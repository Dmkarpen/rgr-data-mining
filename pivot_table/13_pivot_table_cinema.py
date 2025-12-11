import os
from datetime import datetime

import pandas as pd

print("Старт виконання скрипта")

# --------------------------------------
# Шляхи до файлів
# --------------------------------------
DATA_FILE = "cinema_dataset_1000.csv"
RESULTS_DIR = os.path.join("results")
REPORT_FILE = os.path.join(RESULTS_DIR, "cinema_pivot_tables.md")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------
# Завантаження даних
# --------------------------------------
print(f"Спроба завантажити дані з файлу: {DATA_FILE}")

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"ПОМИЛКА: файл {DATA_FILE} не знайдено.")
    print("Переконайтесь, що cinema_dataset_1000.csv знаходиться в поточній папці.")
    raise

print("Дані успішно завантажено")

# --------------------------------------
# Підготовка даних
# --------------------------------------
print("Підготовка даних...")

# Перетворення стовпця date у тип datetime
df["date"] = pd.to_datetime(df["date"])

# Додавання року, номера місяця та назви місяця
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.month_name()

# Обчислення виручки за сеанс (revenue = ticket_price * tickets_sold)
df["revenue"] = df["ticket_price"] * df["tickets_sold"]

print("Підготовка даних завершена")

# --------------------------------------
# Створення зведених таблиць
# --------------------------------------
print("Побудова зведених таблиць...")

# Таблиця 1: середня заповненість залу за фільмами і днями тижня
pivot_occupancy = pd.pivot_table(
    df,
    values="occupancy",
    index="movie_title",
    columns="day_of_week",
    aggfunc="mean"
)

# Упорядкування днів тижня (якщо такі є в даних)
day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
pivot_occupancy = pivot_occupancy.reindex(columns=day_order)

# Таблиця 2: сумарна виручка за місяцями і жанрами
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
df["month_name"] = pd.Categorical(df["month_name"], categories=month_order, ordered=True)

pivot_revenue = pd.pivot_table(
    df,
    values="revenue",
    index="month_name",
    columns="genre",
    aggfunc="sum"
).dropna(how="all")  # видаляємо порожні місяці, якщо таких немає у вибірці

print("Зведені таблиці побудовано")

# --------------------------------------
# Формування простого файлу зі зведеними таблицями
# --------------------------------------
print(f"Формування файлу зі зведеними таблицями: {REPORT_FILE}")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("# Зведені таблиці для аналізу даних кінотеатру\n\n")
    f.write(f"_Дата формування файлу: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n")

    # --- Таблиця 1 ---
    f.write("## Таблиця 1. Середня заповненість залу за фільмами та днями тижня\n\n")
    f.write(
        "У цій зведеній таблиці для кожного фільму (`movie_title`) обчислено "
        "середню заповненість залу (`occupancy`, у відсотках) окремо для кожного дня тижня "
        "(`day_of_week`). Така таблиця може бути використана для загального уявлення про те, "
        "як змінюється заповненість залу залежно від дня тижня для різних фільмів.\n\n"
    )

    try:
        f.write(pivot_occupancy.to_markdown(floatfmt=".2f"))
    except Exception:
        f.write("```\n")
        f.write(pivot_occupancy.to_string(float_format=lambda x: f"{x:.2f}"))
        f.write("\n```\n")

    f.write("\n\n")

    # --- Таблиця 2 ---
    f.write("## Таблиця 2. Сумарна виручка за місяцями та жанрами фільмів\n\n")
    f.write(
        "У цій зведеній таблиці для кожного місяця (`month_name`) та кожного жанру (`genre`) "
        "подано сумарну виручку (`revenue`) від продажу квитків. "
        "Така таблиця може бути використана для загального огляду того, які жанри "
        "приносять більшу або меншу виручку в різні місяці.\n\n"
    )

    try:
        f.write(pivot_revenue.to_markdown(floatfmt=",.2f"))
    except Exception:
        f.write("```\n")
        f.write(pivot_revenue.to_string(float_format=lambda x: f"{x:,.2f}"))
        f.write("\n```\n")

print("Файл зі зведеними таблицями успішно сформовано.")
print("Роботу скрипта завершено.")
