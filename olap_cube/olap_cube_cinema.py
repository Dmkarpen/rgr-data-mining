import os
from datetime import datetime

import pandas as pd


# -----------------------------
# Налаштування
# -----------------------------
DATA_FILE = "cinema_dataset_1000.csv"
RESULTS_DIR = os.path.join("results")
REPORT_FILE = os.path.join(RESULTS_DIR, "05_olap_cube_cinema.md")

os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# Допоміжні функції
# -----------------------------
def df_to_md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """
    Повертає Markdown-таблицю.
    Якщо 'tabulate' не встановлено — повертає текстову таблицю в code block.
    """
    try:
        # to_markdown потребує tabulate (опційна залежність)
        return df.to_markdown(floatfmt=floatfmt)
    except Exception:
        return "```\n" + df.to_string() + "\n```"


def write_section(f, title: str, body: str = "") -> None:
    f.write(f"## {title}\n\n")
    if body:
        f.write(body.strip() + "\n\n")


# -----------------------------
# 1) Завантаження даних
# -----------------------------
print(f"Завантаження даних з файлу: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

required_cols = ["date", "movie_title", "genre", "day_of_week", "ticket_price", "tickets_sold", "occupancy"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"У датасеті відсутні необхідні колонки: {missing}")

print("Дані завантажено успішно.")


# -----------------------------
# 2) Підготовка даних і базові міри
# -----------------------------
print("Підготовка даних та розрахунок мір...")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
if df["date"].isna().any():
    raise ValueError("Виявлено некоректні значення у колонці 'date' (не вдалося конвертувати у datetime).")

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["revenue"] = df["ticket_price"] * df["tickets_sold"]

# Базові виміри (Dimensions): year, month, day_of_week, movie_title, genre
dims = ["year", "month", "day_of_week", "movie_title", "genre"]

# Базові міри (Measures):
# - total_tickets_sold = SUM(tickets_sold)
# - total_revenue      = SUM(revenue)
# - avg_occupancy      = AVG(occupancy)
print("Формування OLAP-куба (агрегація)...")

cube = (
    df.groupby(dims, dropna=False)
      .agg(
          total_tickets_sold=("tickets_sold", "sum"),
          total_revenue=("revenue", "sum"),
          avg_occupancy=("occupancy", "mean")
      )
      .reset_index()
)

# Куб як DataFrame з MultiIndex (для OLAP-операцій)
cube_mi = cube.set_index(dims).sort_index()

print("OLAP-куб сформовано.")
print(f"Кількість комбінацій у кубі: {len(cube_mi)}")


# -----------------------------
# 3) OLAP-операції
# -----------------------------
# Виберемо “еталонні” значення з даних, щоб приклади гарантовано працювали
sample_year = int(df["year"].min())
sample_movie = str(df["movie_title"].dropna().unique()[0])

# 3.1 SLICE: фіксуємо (year=sample_year, movie_title=sample_movie) і дивимось розріз по іншим вимірам
print("Операція SLICE...")
slice_df = cube_mi.xs((sample_year, sample_movie), level=("year", "movie_title"), drop_level=False)

# 3.2 DRILL-DOWN: деталізація часу
# Приклад: беремо обраний рік -> показуємо month x day_of_week по виручці (SUM)
print("Операція DRILL-DOWN (year -> month -> day_of_week)...")
drill_df = (
    df[df["year"] == sample_year]
    .groupby(["month", "day_of_week"])
    .agg(total_revenue=("revenue", "sum"), total_tickets_sold=("tickets_sold", "sum"))
    .reset_index()
)

# Перетворимо на зведену форму: rows=month, cols=day_of_week, values=total_revenue
day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
drill_pivot = pd.pivot_table(
    drill_df,
    values="total_revenue",
    index="month",
    columns="day_of_week",
    aggfunc="sum",
    fill_value=0
).reindex(columns=day_order)

# 3.3 ROLL-UP: згортання (агрегація) до більш високого рівня
# Приклад: підсумок по року і жанру (SUM виручки, SUM квитків, AVG заповненості)
print("Операція ROLL-UP (до рівня year x genre)...")
rollup_df = (
    df.groupby(["year", "genre"])
      .agg(
          total_revenue=("revenue", "sum"),
          total_tickets_sold=("tickets_sold", "sum"),
          avg_occupancy=("occupancy", "mean")
      )
      .reset_index()
      .sort_values(["year", "total_revenue"], ascending=[True, False])
)

# 3.4 ROTATE / PIVOT: “обертання” куба
# Приклад: рядки = movie_title, стовпці = day_of_week, значення = AVG occupancy
print("Операція ROTATE/PIVOT (movie_title x day_of_week -> avg occupancy)...")
rotate_df = pd.pivot_table(
    df,
    values="occupancy",
    index="movie_title",
    columns="day_of_week",
    aggfunc="mean"
).reindex(columns=day_order)


# -----------------------------
# 4) Формування Markdown-файлу з результатами
# -----------------------------
print(f"Формування Markdown-файлу: {REPORT_FILE}")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("# OLAP-куб для даних кінотеатру та базові OLAP-операції\n\n")

    write_section(
        f,
        "Використані виміри та міри",
        (
            "**Виміри (Dimensions):** `year`, `month`, `day_of_week`, `movie_title`, `genre`.\n\n"
            "**Міри (Measures):**\n"
            "- `total_tickets_sold` — сумарна кількість проданих квитків (SUM)\n"
            "- `total_revenue` — сумарна виручка (SUM), де `revenue = ticket_price × tickets_sold`\n"
            "- `avg_occupancy` — середня заповненість залу (AVG)\n"
        ),
    )

    # Фрагмент куба (перші N рядків)
    preview = cube_mi.reset_index().head(20)
    write_section(
        f,
        "Фрагмент OLAP-куба (перші 20 рядків)",
        "Нижче наведено приклад структури куба після агрегації за вимірами.",
    )
    f.write(df_to_md_table(preview, floatfmt=".2f") + "\n\n")

    # SLICE
    write_section(
        f,
        "OLAP-операція: SLICE (зріз)",
        (
            f"Зріз виконується шляхом фіксації окремих значень вимірів.\n\n"
            f"У цьому прикладі зафіксовано:\n"
            f"- `year = {sample_year}`\n"
            f"- `movie_title = \"{sample_movie}\"`\n\n"
            f"Результат показує агреговані значення для обраного фільму в обраному році "
            f"у розрізі інших вимірів (місяць, день тижня, жанр)."
        ),
    )
    f.write(df_to_md_table(slice_df.reset_index().head(30), floatfmt=".2f") + "\n\n")

    # DRILL-DOWN
    write_section(
        f,
        "OLAP-операція: DRILL-DOWN (деталізація)",
        (
            f"Деталізація (drill-down) дозволяє перейти від більш загального рівня до детальнішого.\n\n"
            f"У цьому прикладі для `year = {sample_year}` виконано деталізацію за часом:\n"
            f"`year → month → day_of_week`.\n\n"
            f"Нижче наведено зведену таблицю виручки (SUM) за місяцями та днями тижня."
        ),
    )
    f.write(df_to_md_table(drill_pivot, floatfmt=",.2f") + "\n\n")

    # ROLL-UP
    write_section(
        f,
        "OLAP-операція: ROLL-UP (агрегація / згортання)",
        (
            "Агрегація (roll-up) дозволяє піднятися на вищий рівень узагальнення.\n\n"
            "У цьому прикладі дані згорнуто до рівня `year × genre` з такими мірами:\n"
            "- `total_revenue` (SUM)\n"
            "- `total_tickets_sold` (SUM)\n"
            "- `avg_occupancy` (AVG)\n"
        ),
    )
    f.write(df_to_md_table(rollup_df.head(30), floatfmt=",.2f") + "\n\n")

    # ROTATE / PIVOT
    write_section(
        f,
        "OLAP-операція: ROTATE / PIVOT (обертання вимірів)",
        (
            "Операція rotate/pivot змінює представлення даних: виміри можуть мінятися місцями між "
            "рядками та стовпцями.\n\n"
            "У цьому прикладі побудовано зведену таблицю, де:\n"
            "- рядки: `movie_title`\n"
            "- стовпці: `day_of_week`\n"
            "- значення: середня заповненість `avg(occupancy)`\n"
        ),
    )
    f.write(df_to_md_table(rotate_df, floatfmt=".2f") + "\n\n")

print("Готово. Markdown-файл сформовано.")
print(f"Шлях до файлу: {REPORT_FILE}")
