import os
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

DATA_FILE = "cinema_dataset_1000.csv"
RESULTS_DIR = "results"

REPORT_FILE = os.path.join(RESULTS_DIR, "12_network_analysis_cinema_report.md")
PLOT_FILE = os.path.join(RESULTS_DIR, "12_movie_day_graph.png")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =======================================
# 1. Завантаження даних
# =======================================
print(f"Завантаження даних із файлу: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

required_cols = ["movie_title", "day_of_week", "tickets_sold"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"У наборі даних відсутня необхідна колонка: '{col}'")

# =======================================
# 2. Побудова двочасткового графа Movie <-> DayOfWeek
# =======================================
print("\nПобудова двочасткового графа 'Movie <-> DayOfWeek'")

G = nx.Graph()

movie_nodes = sorted(df["movie_title"].unique())
day_nodes = sorted(df["day_of_week"].unique())

G.add_nodes_from(movie_nodes, bipartite=0, type="movie")
G.add_nodes_from(day_nodes, bipartite=1, type="day")

edge_weights_df = (
    df.groupby(["movie_title", "day_of_week"])["tickets_sold"]
      .sum()
      .reset_index(name="weight")
)

weighted_edges = [
    (row["movie_title"], row["day_of_week"], row["weight"])
    for _, row in edge_weights_df.iterrows()
]

G.add_weighted_edges_from(weighted_edges)

print("Граф побудовано.")
print(f"Кількість вершин: {G.number_of_nodes()}")
print(f"Кількість ребер: {G.number_of_edges()}")

# =======================================
# 3. Аналіз графа (алгоритми обходу)
# =======================================
print("\nАналіз графа за допомогою алгоритмів обходу")

components = list(nx.connected_components(G))
components_result = f"У графі знайдено {len(components)} компонент(и) зв'язності."
print(components_result)

movie_list = movie_nodes
shortest_path_text = ""

if len(movie_list) >= 2:
    start_movie = movie_list[0]
    end_movie = movie_list[-1]
    try:
        length = nx.shortest_path_length(G, source=start_movie, target=end_movie)
        shortest_path_text = (
            f"Найкоротший шлях між фільмами '{start_movie}' "
            f"та '{end_movie}' має довжину {length} крок(ів).\n\n"
            "Це означає, що вони пов'язані через спільні дні показу "
            "або через інші фільми й дні у двочастковій мережі.\n"
        )
        print(
            f"Найкоротший шлях між фільмами '{start_movie}' "
            f"та '{end_movie}': {length} крок(ів)."
        )
    except nx.NetworkXNoPath:
        shortest_path_text = (
            f"Між фільмами '{start_movie}' та '{end_movie}' "
            "відсутній шлях у даному графі.\n"
        )
        print(
            f"Між фільмами '{start_movie}' та '{end_movie}' "
            "немає шляху."
        )
else:
    shortest_path_text = (
        "У наборі даних недостатньо фільмів для обчислення найкоротшого шляху.\n"
    )
    print("Недостатньо фільмів для обчислення найкоротшого шляху.")

# =======================================
# 4. Візуалізація графа
# =======================================
print(f"\nЗбереження зображення графа у файл: {PLOT_FILE}")

plt.figure(figsize=(14, 10))

pos = {}
pos.update((node, (0, i)) for i, node in enumerate(movie_nodes))
pos.update((node, (1, i * 0.8)) for i, node in enumerate(day_nodes))

nx.draw_networkx_nodes(
    G, pos, nodelist=movie_nodes,
    node_color="skyblue", node_size=2000, label="Movies"
)
nx.draw_networkx_nodes(
    G, pos, nodelist=day_nodes,
    node_color="lightgreen", node_size=2000, label="Days of week"
)

edge_weights = nx.get_edge_attributes(G, "weight")
if edge_weights:
    max_weight = max(edge_weights.values())
    widths = [edge_weights[e] / max_weight * 4 + 0.5 for e in G.edges()]
else:
    widths = 1.0

nx.draw_networkx_edges(G, pos, width=widths, alpha=0.7, edge_color="gray")
nx.draw_networkx_labels(G, pos, font_size=9)

plt.title(
    "Двочастковий граф 'Movie <-> DayOfWeek' "
    "(товщина ребра = загальна кількість проданих квитків)",
    fontsize=14
)
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_FILE)
plt.close()
print("Зображення графа збережено.")

# =======================================
# 5. Зведена таблиця
# =======================================
pivot = (
    edge_weights_df
    .pivot_table(
        index="movie_title",
        columns="day_of_week",
        values="weight",
        aggfunc="sum",
        fill_value=0
    )
)

pivot_md = pivot.to_markdown()

# =======================================
# 6. Формування Markdown-звіту
# =======================================
print(f"\nФормування текстового звіту: {REPORT_FILE}")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("# Мережевий аналіз набору даних кінотеатру (СР №12)\n\n")

    f.write("## 1. Набір даних та побудова графа\n\n")
    f.write(
        "Для дослідження використано набір даних `cinema_dataset_1000.csv`. "
        "Були визначені два типи вершин:\n\n"
        "- **Фільми** (`movie_title`)\n"
        "- **Дні тижня** (`day_of_week`)\n\n"
        "На їх основі побудовано неорієнтований ваговий двочастковий граф, де:\n\n"
        "- вершини першої частки відповідають фільмам,\n"
        "- вершини другої частки відповідають дням тижня,\n"
        "- ребро між ними існує, якщо фільм демонструвався у відповідний день,\n"
        "- вага ребра — загальна кількість проданих квитків на цей фільм у цей день.\n\n"
    )
    f.write(
        f"У підсумку граф містить **{G.number_of_nodes()} вершин** та "
        f"**{G.number_of_edges()} ребер**.\n\n"
    )

    f.write("## 2. Аналіз за допомогою алгоритмів обходу (DFS / BFS)\n\n")

    f.write("### 2.1. Компоненти зв’язності (DFS)\n\n")
    f.write(components_result + "\n\n")
    if len(components) > 1:
        f.write(
            "Наявність декількох компонент свідчить, що частина фільмів або днів тижня "
            "не має зв’язків з іншими елементами розкладу.\n\n"
        )
    else:
        f.write(
            "Усі фільми та дні тижня утворюють єдину взаємопов’язану мережу, "
            "що вказує на повну зв’язність графа.\n\n"
        )

    f.write("### 2.2. Найкоротший шлях між фільмами (BFS)\n\n")
    f.write(shortest_path_text + "\n")

    f.write("## 3. Візуалізація графа\n\n")
    f.write(
        "Нижче наведено зображення двочасткового графа, де товщина ребер "
        "відповідає кількості проданих квитків:\n\n"
    )
    f.write("![Двочастковий граф Movie-DayOfWeek](12_movie_day_graph.png)\n\n")

    f.write("## 4. Зведена таблиця (аналог OLAP-куба)\n\n")
    f.write(
        "У таблиці нижче наведено сумарну кількість проданих квитків для кожного фільму "
        "у розрізі днів тижня. Рядки — фільми, стовпці — дні тижня:\n\n"
    )
    f.write(pivot_md + "\n\n")

    f.write("## 5. Висновки\n\n")
    f.write(
        "- Створена мережа дає змогу аналізувати взаємозв’язок між фільмами "
        "та днями тижня за кількістю продажів квитків.\n"
        "- Алгоритми DFS і BFS дозволяють визначати структуру графа та досліджувати "
        "зв’язність між різними елементами розкладу.\n"
        "- Вагові ребра дозволяють виявити найбільш прибуткові дні для кожного фільму "
        "та визначити закономірності продажів.\n"
        "- Зведена таблиця може бути використана як інструмент для подальшого "
        "аналітичного опрацювання, зокрема оптимізації розкладу та цінової політики.\n"
    )

print("Формування Markdown-звіту завершено.")
