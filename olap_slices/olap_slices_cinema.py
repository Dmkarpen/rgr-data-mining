import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Налаштування
# =========================
DATA_FILE = "cinema_dataset_1000.csv"

RESULTS_DIR = os.path.join("results")
REPORT_MD = os.path.join(RESULTS_DIR, "olap_slices_cinema_report.md")

IMG_STRATEGIC = os.path.join(RESULTS_DIR, "01_strategic_revenue_by_year_genre.png")
IMG_KPI = os.path.join(RESULTS_DIR, "02_kpi_movies_scatter.png")
IMG_HEATMAP = os.path.join(RESULTS_DIR, "03_heatmap_occupancy_movie_day.png")

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# =========================
# Допоміжні функції
# =========================
def df_to_md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """
    Повертає таблицю у Markdown.
    Якщо 'tabulate' не встановлено — повертає таблицю як текст у code block.
    """
    try:
        return df.to_markdown(floatfmt=floatfmt)
    except Exception:
        return "```\n" + df.to_string() + "\n```"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class CinemaDatasetSchema:
    date: str = "date"
    movie_title: str = "movie_title"
    genre: str = "genre"
    day_of_week: str = "day_of_week"
    ticket_price: str = "ticket_price"
    tickets_sold: str = "tickets_sold"
    occupancy: str = "occupancy"


class CinemaDataWarehouse:
    """
    Простий 'Data Warehouse' у стилі Singleton:
    завантажує та готує дані один раз, щоб далі будувати OLAP-зрізи.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CinemaDataWarehouse, cls).__new__(cls)
        return cls._instance

    def __init__(self, csv_path: str, schema: CinemaDatasetSchema = CinemaDatasetSchema()):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.csv_path = csv_path
        self.schema = schema
        self.df = self._load_and_prepare()

    def _load_and_prepare(self) -> pd.DataFrame:
        print(f"Завантаження даних з файлу: {self.csv_path}")

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Файл не знайдено: {self.csv_path}. Переконайтесь, що CSV знаходиться в цій самій папці."
            )

        df = pd.read_csv(self.csv_path)

        required = [
            self.schema.date,
            self.schema.movie_title,
            self.schema.genre,
            self.schema.day_of_week,
            self.schema.ticket_price,
            self.schema.tickets_sold,
            self.schema.occupancy,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"У датасеті відсутні необхідні колонки: {missing}")

        df[self.schema.date] = pd.to_datetime(df[self.schema.date], errors="coerce")
        if df[self.schema.date].isna().any():
            raise ValueError("Виявлено некоректні значення у колонці 'date' (не вдалося конвертувати у datetime).")

        df["year"] = df[self.schema.date].dt.year
        df["month"] = df[self.schema.date].dt.month
        df["revenue"] = df[self.schema.ticket_price] * df[self.schema.tickets_sold]

        # Нормалізація day_of_week у стандартний порядок (якщо є інші значення — залишаться, але не зламають код)
        if self.schema.day_of_week in df.columns:
            df[self.schema.day_of_week] = df[self.schema.day_of_week].astype(str)

        print("Дані успішно завантажено та підготовлено.")
        print(f"Кількість рядків: {len(df)}")
        return df


# =========================
# OLAP-зрізи (ПР-06)
# =========================
def build_strategic_slice(dw: CinemaDataWarehouse) -> pd.DataFrame:
    """
    Стратегічний зріз: виручка за роками та жанрами (SUM).
    """
    df = dw.df
    strategic = (
        df.groupby(["year", dw.schema.genre])["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_revenue"})
    )
    return strategic


def plot_strategic_revenue_by_year_genre(strategic_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Графік: stacked bar для виручки (year x genre).
    Повертає pivot-таблицю, щоб вставити у звіт.
    """
    pivot = pd.pivot_table(
        strategic_df,
        values="total_revenue",
        index="year",
        columns="genre",
        aggfunc="sum",
        fill_value=0
    ).sort_index()

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Стратегічний зріз: виручка за роками та жанрами (SUM)")
    plt.xlabel("Рік")
    plt.ylabel("Виручка (грн)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return pivot


def build_kpi_slice(dw: CinemaDataWarehouse) -> pd.DataFrame:
    """
    Операційний зріз (KPI): показники по фільмах.
    """
    df = dw.df
    kpi = (
        df.groupby([dw.schema.movie_title, dw.schema.genre])
        .agg(
            total_revenue=("revenue", "sum"),
            total_tickets_sold=(dw.schema.tickets_sold, "sum"),
            avg_occupancy=(dw.schema.occupancy, "mean"),
            avg_ticket_price=(dw.schema.ticket_price, "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    return kpi


def plot_kpi_scatter(kpi_df: pd.DataFrame, output_path: str) -> None:
    """
    Scatter: X = avg_occupancy, Y = total_revenue, розмір точки = total_tickets_sold.
    """
    # Щоб розміри були читабельні
    sizes = kpi_df["total_tickets_sold"].clip(lower=1)
    sizes = (sizes / sizes.max()) * 900 + 80  # масштаб

    plt.figure(figsize=(10, 6))
    plt.scatter(kpi_df["avg_occupancy"], kpi_df["total_revenue"], s=sizes, alpha=0.7)
    plt.title("Операційний зріз (KPI): заповненість vs виручка (розмір = продані квитки)")
    plt.xlabel("Середня заповненість (%)")
    plt.ylabel("Сумарна виручка (грн)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Підписи для топ-5 фільмів за виручкою
    top = kpi_df.head(5)
    for _, row in top.iterrows():
        plt.text(row["avg_occupancy"], row["total_revenue"], str(row["movie_title"]), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_product_slice(dw: CinemaDataWarehouse) -> pd.DataFrame:
    """
    Продуктовий зріз: матриця movie_title x day_of_week зі середньою заповненістю.
    """
    df = dw.df
    heat = pd.pivot_table(
        df,
        values=dw.schema.occupancy,
        index=dw.schema.movie_title,
        columns=dw.schema.day_of_week,
        aggfunc="mean"
    )

    # Упорядкуємо дні тижня
    existing_cols = [c for c in DAY_ORDER if c in heat.columns]
    remaining = [c for c in heat.columns if c not in DAY_ORDER]
    heat = heat[existing_cols + remaining]

    # Сортування фільмів за середнім значенням (опційно, для читабельності)
    heat["__avg__"] = heat.mean(axis=1)
    heat = heat.sort_values("__avg__", ascending=False).drop(columns=["__avg__"])

    return heat


def plot_heatmap(heat_df: pd.DataFrame, output_path: str) -> None:
    """
    Heatmap без seaborn: matplotlib imshow.
    """
    plt.figure(figsize=(12, max(5, 0.45 * len(heat_df))))
    data = heat_df.fillna(0).values

    plt.imshow(data, aspect="auto")
    plt.title("Продуктовий зріз: середня заповненість (%) за фільмами та днями тижня")
    plt.xlabel("День тижня")
    plt.ylabel("Фільм")

    plt.xticks(range(len(heat_df.columns)), heat_df.columns, rotation=0)
    plt.yticks(range(len(heat_df.index)), heat_df.index)

    cbar = plt.colorbar()
    cbar.set_label("Заповненість (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# =========================
# Генерація Markdown-звіту
# =========================
def generate_markdown_report(
    output_path: str,
    strategic_pivot: pd.DataFrame,
    kpi_df: pd.DataFrame,
    heat_df: pd.DataFrame,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Практична робота №6: OLAP-зрізи та візуалізація для даних кінотеатру\n\n")

        f.write("## 1. Вихідні дані\n\n")
        f.write(
            "Для аналізу використано набір даних кінотеатру у форматі CSV. "
            "Дані містять інформацію про дату сеансу, фільм, жанр, день тижня, ціну квитка, "
            "кількість проданих квитків та заповненість залу.\n\n"
            "Додатково обчислено показник **виручки**:\n\n"
            "- `revenue = ticket_price × tickets_sold`\n\n"
        )

        # ---- Стратегічний зріз ----
        f.write("## 2. Стратегічний зріз\n\n")
        f.write(
            "**Мета зрізу:** оцінити структуру виручки кінотеатру в розрізі **років** та **жанрів**.\n\n"
            "- Виміри: `year`, `genre`\n"
            "- Міра: `SUM(revenue)`\n\n"
        )
        f.write("### Таблиця 2.1. Виручка за роками та жанрами (pivot)\n\n")
        f.write(df_to_md_table(strategic_pivot, floatfmt=",.2f") + "\n\n")
        f.write("### Рисунок 1. Стратегічний зріз: виручка за роками та жанрами\n\n")
        f.write(f"![Рисунок 1 – Виручка за роками та жанрами]({os.path.basename(IMG_STRATEGIC)})\n\n")
        f.write(
            "Пояснення: графік показує, як змінюється сумарна виручка у різні роки, "
            "а також який внесок у загальну виручку робить кожен жанр.\n\n"
        )

        # ---- KPI-зріз ----
        f.write("## 3. Операційний зріз (KPI)\n\n")
        f.write(
            "**Мета зрізу:** порівняти ефективність фільмів за ключовими показниками.\n\n"
            "- Виміри: `movie_title`, `genre`\n"
            "- Міри: `SUM(revenue)`, `SUM(tickets_sold)`, `AVG(occupancy)`, `AVG(ticket_price)`\n\n"
        )
        f.write("### Таблиця 3.1. KPI-показники по фільмах (TOP-15 за виручкою)\n\n")
        f.write(df_to_md_table(kpi_df.head(15), floatfmt=",.2f") + "\n\n")
        f.write("### Рисунок 2. KPI-зріз: заповненість vs виручка (розмір точки = продані квитки)\n\n")
        f.write(f"![Рисунок 2 – KPI-зріз по фільмах]({os.path.basename(IMG_KPI)})\n\n")
        f.write(
            "Пояснення: точкова діаграма демонструє взаємозв’язок між середньою заповненістю залу та виручкою. "
            "Розмір точки відображає сумарну кількість проданих квитків. "
            "Це дозволяє швидко знаходити фільми з високою/низькою ефективністю.\n\n"
        )

        # ---- Продуктовий зріз ----
        f.write("## 4. Продуктовий зріз\n\n")
        f.write(
            "**Мета зрізу:** дослідити середню заповненість залу у розрізі **фільмів** та **днів тижня**.\n\n"
            "- Рядки: `movie_title`\n"
            "- Стовпці: `day_of_week`\n"
            "- Значення: `AVG(occupancy)`\n\n"
        )
        f.write("### Таблиця 4.1. Матриця заповненості (AVG occupancy)\n\n")
        f.write(df_to_md_table(heat_df, floatfmt=".2f") + "\n\n")
        f.write("### Рисунок 3. Теплова карта заповненості (фільм × день тижня)\n\n")
        f.write(f"![Рисунок 3 – Теплова карта заповненості]({os.path.basename(IMG_HEATMAP)})\n\n")
        f.write(
            "Пояснення: теплова карта відображає середню заповненість для кожного фільму залежно від дня тижня. "
            "Це зручно для порівняння патернів відвідуваності та подальшої оптимізації розкладу.\n\n"
        )

        f.write("## 5. Підсумок\n\n")
        f.write(
            "У межах практичної роботи побудовано три базові OLAP-зрізи (стратегічний, операційний та продуктовий), "
            "а також виконано їх візуалізацію. Отримані таблиці та графіки можуть використовуватися як основа "
            "для подальшої аналітики та управлінських рішень у рамках проєкту «Кінотеатр».\n"
        )


# =========================
# main
# =========================
def main() -> None:
    ensure_dir(RESULTS_DIR)

    dw = CinemaDataWarehouse(DATA_FILE)

    print("\n=== Побудова OLAP-зрізів (ПР-06) ===")

    # 1) Стратегічний зріз
    strategic = build_strategic_slice(dw)
    strategic_pivot = plot_strategic_revenue_by_year_genre(strategic, IMG_STRATEGIC)
    print(f"Стратегічний зріз сформовано. Зображення збережено: {IMG_STRATEGIC}")

    # 2) KPI-зріз
    kpi = build_kpi_slice(dw)
    plot_kpi_scatter(kpi, IMG_KPI)
    print(f"KPI-зріз сформовано. Зображення збережено: {IMG_KPI}")

    # 3) Продуктовий зріз (heatmap)
    heat = build_product_slice(dw)
    plot_heatmap(heat, IMG_HEATMAP)
    print(f"Продуктовий зріз сформовано. Зображення збережено: {IMG_HEATMAP}")

    # Markdown-звіт
    generate_markdown_report(REPORT_MD, strategic_pivot, kpi, heat)
    print(f"\nMarkdown-звіт сформовано: {REPORT_MD}")
    print("Роботу завершено успішно.")


if __name__ == "__main__":
    main()
