from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional

Fact = Tuple[str, str]  # (атрибут, значення)

@dataclass
class Rule:
    """Правило «ЯКЩО ... ТО ...»"""
    name: str
    conditions: List[Fact]
    action: Callable[[Dict[str, str], List[str]], None]
    priority: int = 0

    def is_active(self, state: Dict[str, str]) -> bool:
        return all(state.get(k) == v for k, v in self.conditions)

@dataclass
class Engine:
    """Рушій прямого виведення з розв'язанням конфліктів."""
    state: Dict[str, str]
    rules: List[Rule]
    fired: List[str] = field(default_factory=list)          # імена правил, що вже спрацювали
    applied_actions: List[str] = field(default_factory=list)  # тексти застосованих дій

    def match_rules(self) -> List[Rule]:
        return [r for r in self.rules if r.is_active(self.state) and r.name not in self.fired]

    def resolve_conflict(self, active: List[Rule]) -> Optional[Rule]:
        if not active:
            return None
        active.sort(key=lambda r: (r.priority, len(r.conditions)), reverse=True)
        return active[0]

    def run(self, max_steps: int = 50) -> None:
        for _ in range(max_steps):
            active = self.match_rules()
            if not active:
                break
            rule = self.resolve_conflict(active)
            if rule is None:
                break
            before = dict(self.state)
            rule.action(self.state, self.applied_actions)
            self.fired.append(rule.name)
            if self.state == before and (not self.applied_actions or self.applied_actions[-1] == ""):
                break

# ---------- Утиліти дій ----------
def add_recommendation(state: Dict[str, str], text: str):
    recs = state.get("рекомендації", "")
    parts = [p.strip() for p in recs.split(";") if p.strip()]
    if text not in parts:
        parts.append(text)
    state["рекомендації"] = "; ".join(parts)

def action_recommend(text: str):
    def _do(state: Dict[str, str], applied: List[str]):
        add_recommendation(state, text)
        if text not in applied:
            applied.append(text)
    return _do

# ---------- Правила ----------
rules: List[Rule] = [
    Rule(
        name="R1:weekday_daytime_price_cut",
        conditions=[("день", "Пн/Вт/Ср/Чт"), ("час", "<=16")],
        action=action_recommend("Знизити ціну на ~10% для підвищення попиту"),
        priority=5,
    ),
    Rule(
        name="R2:weekend_evening_comedy_hold",
        conditions=[("день", "Сб/Нд"), ("час", ">=18"), ("жанр", "комедія")],
        action=action_recommend("Тримати ціну: очікується високий попит"),
        priority=6,
    ),
    Rule(
        name="R3:horror_drama_shift",
        conditions=[("жанр", "жахи/драма"), ("час", "<18")],
        action=action_recommend("Змістити показ на >=18:00"),
        priority=4,
    ),
    Rule(
        name="R4:price_capacity_promo",
        conditions=[("ціна", "висока"), ("місткість", "велика")],
        action=action_recommend("Увімкнути промо-акцію (2×1 або -15%)"),
        priority=3,
    ),
    Rule(
        name="R5:promo_then_shift",
        conditions=[("день", "Пн/Вт/Ср/Чт"), ("час", "<=16"), ("промо", "увімкнено")],
        action=action_recommend("Крім промо, змістити початок на >=18:00"),
        priority=7,
    ),
]

def postprocess(engine: Engine):
    if any("промо-акцію" in a for a in engine.applied_actions):
        engine.state["промо"] = "увімкнено"

def pretty_print_state(title: str, state: Dict[str, str], fired: List[str], actions: List[str]):
    print(title)
    print("-" * len(title))
    order = ["день", "час", "жанр", "ціна", "місткість", "промо", "рекомендації"]
    for k in order:
        if k in state:
            print(f"{k:>12}: {state[k]}")
    for k, v in state.items():
        if k not in order:
            print(f"{k:>12}: {v}")
    print(f"Спрацьовані правила: {', '.join(fired) if fired else '(немає)'}")
    if actions:
        print("Застосовані дії:")
        for i, a in enumerate(actions, 1):
            print(f"  {i:>2}. {a}")
    print()

# ---------- Демонстрація ----------
if __name__ == "__main__":
    case1 = {
        "день": "Пн/Вт/Ср/Чт",
        "час": "<=16",
        "жанр": "жахи/драма",
        "ціна": "висока",
        "місткість": "велика",
        "промо": "вимкнено",
    }
    eng1 = Engine(state=case1, rules=rules)
    eng1.run()
    postprocess(eng1)
    eng1.run()
    pretty_print_state("Приклад 1: будній/14:00/жахи/висока ціна/великий зал", eng1.state, eng1.fired, eng1.applied_actions)

    case2 = {
        "день": "Сб/Нд",
        "час": ">=18",
        "жанр": "комедія",
        "ціна": "середня",
        "місткість": "середня",
        "промо": "вимкнено",
    }
    eng2 = Engine(state=case2, rules=rules)
    eng2.run()
    pretty_print_state("Приклад 2: вихідні/20:00/комедія/середня ціна/середній зал", eng2.state, eng2.fired, eng2.applied_actions)
