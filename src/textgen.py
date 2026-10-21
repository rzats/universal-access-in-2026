"""Ukrainian text content for synthetic signage, packaging and document stimuli.

Every generated line is a list of (segment_text, category) pieces so that each
ground-truth token keeps the category of the slot it came from:

  word   common vocabulary (shop types, product names, notice phrases)
  name   real proper names (streets, cities, people, drug names)
  nonce  invented words a language prior cannot predict
  lure   a real word with one letter changed (tests normalisation to the prior)
  num    digits (dates, prices, phones, doses); assigned automatically per token

The "name slot" of every stimulus (brand / shop / company / surname) is set by the
caller to one of {word, nonce, lure} so the three conditions are balanced.
"""
import random

VOWELS = "аеиіоуяюєї"
CONS = "бвгґджзклмнпрстфхцчшщ"

SHOP_TYPES = ["Аптека", "Кав'ярня", "Пекарня", "Перукарня", "Квіти", "Книгарня", "Оптика",
              "Хімчистка", "Продукти", "Стоматологія", "Ветеринарна клініка", "Ремонт взуття",
              "Канцтовари", "Овочі та фрукти", "Господарчі товари", "Ательє", "Піцерія", "Їдальня"]
# real words that are typical of Ukrainian brand / shop names (the 'word' condition)
BRAND_WORDS = ["Ластівка", "Ромашка", "Калина", "Добробут", "Барвінок", "Світанок", "Волошка",
               "Джерело", "Яблунька", "Веселка", "Колосок", "Садочок", "Медунка", "Смерічка",
               "Лелека", "Криниця", "Затишок", "Подорожник", "Горобина", "Жайворонок", "Злагода",
               "Вишенька", "Соняшник", "Берегиня", "Карпати", "Дніпро", "Полісся", "Ґражда",
               "Їжачок", "Євшан", "Мрія", "Родина", "Гостинець", "Сузір'я", "Оберіг", "Перлина"]
STREETS = ["Городоцька", "Личаківська", "Зелена", "Стрийська", "Шевченка", "Івана Франка",
           "Січових Стрільців", "Богдана Хмельницького", "Героїв Майдану", "Ґонти", "Лесі Українки",
           "Князя Романа", "Пекарська", "Коперника", "Дорошенка", "Кульпарківська", "Наукова",
           "Сихівська", "Хвильового", "Володимира Великого", "Митрополита Шептицького",
           "Пастернака", "Ярослава Мудрого", "Соборна", "Замкова", "Ґалаґана", "Їжакевича",
           "Євгена Коновальця", "Тиха", "Садова"]
CITIES = ["Львів", "Київ", "Житомир", "Вінниця", "Тернопіль", "Чернівці", "Івано-Франківськ",
          "Рівне", "Луцьк", "Ужгород", "Полтава", "Черкаси", "Хмельницький", "Біла Церква",
          "Дрогобич", "Кременчук", "Коломия", "Ніжин"]
FIRST_F = ["Ольга", "Ірина", "Оксана", "Наталія", "Галина", "Марія", "Юлія", "Софія", "Тетяна",
           "Христина", "Євгенія", "Ярина"]
FIRST_M = ["Андрій", "Тарас", "Василь", "Іван", "Олег", "Богдан", "Юрій", "Михайло", "Ярослав",
           "Святослав", "Євген", "Ігор"]
PATRON = [("Іванівна", "Іванович"), ("Петрівна", "Петрович"), ("Василівна", "Васильович"),
          ("Михайлівна", "Михайлович"), ("Степанівна", "Степанович"), ("Ігорівна", "Ігорович"),
          ("Ярославівна", "Ярославович"), ("Олегівна", "Олегович")]
SURNAMES = ["Коваленко", "Шевчук", "Бондаренко", "Ткаченко", "Кравчук", "Мельник", "Савчук",
            "Гнатюк", "Дячук", "Пономаренко", "Лисенко", "Гуменюк", "Ґудзь", "Стецько", "Яремчук",
            "Возняк", "Костюк", "Федорів", "Рибак", "Їжак"]
SURNAME_SUFFIX = ["енко", "чук", "ишин", "ук", "ів", "юк"]
MONTHS = ["січень", "лютий", "березень", "квітень", "травень", "червень", "липень", "серпень",
          "вересень", "жовтень", "листопад", "грудень"]

NOTICES = [
    ["Зачинено на санітарний день"],
    ["Вхід з двору"],
    ["Ліфт не працює до ", ("date",)],
    ["Обережно, слизька підлога"],
    ["Найближче укриття праворуч, ", ("int", 20, 300), " м"],
    ["Технічна перерва з ", ("time",), " до ", ("time",)],
    ["Прийом посилок у вікні № ", ("int", 1, 9)],
    ["Паркування заборонено"],
    ["Не зачиняти двері"],
    ["Працюємо без вихідних"],
    ["Оплата карткою не приймається"],
    ["Відключення гарячої води ", ("date",)],
    ["Зачинено на переоблік ", ("date",)],
    ["Вільних місць немає"],
]

FOODS = [  # (product, unit kind, ingredient pool key)
    ("Молоко пастеризоване", "l", "milk"), ("Кефір", "l", "milk"), ("Ряжанка", "l", "milk"),
    ("Сир кисломолочний", "g", "milk"), ("Сметана", "g", "milk"), ("Масло вершкове", "g", "milk"),
    ("Йогурт полуничний", "g", "milk"), ("Хліб житній", "g", "bread"), ("Батон нарізний", "g", "bread"),
    ("Печиво вівсяне", "g", "sweet"), ("Пряники медові", "g", "sweet"), ("Шоколад чорний", "g", "sweet"),
    ("Ковбаса варена", "g", "meat"), ("Сосиски молочні", "g", "meat"), ("Вареники з картоплею", "g", "dough"),
    ("Пельмені домашні", "g", "dough"), ("Сік яблучний", "l", "juice"), ("Нектар персиковий", "l", "juice"),
]
INGR = {
    "milk": ["молоко коров'яче", "закваска", "вершки", "полуниця", "цукор", "пектин", "сіль"],
    "bread": ["борошно житнє", "борошно пшеничне", "вода питна", "дріжджі", "сіль", "солод", "олія соняшникова"],
    "sweet": ["вівсяні пластівці", "борошно пшеничне", "цукор", "мед", "какао", "соєвий лецитин", "маргарин", "яйця курячі"],
    "meat": ["свинина", "яловичина", "вода питна", "крохмаль", "сіль", "спеції", "часник", "молоко сухе"],
    "dough": ["борошно пшеничне", "картопля", "цибуля", "вода питна", "олія соняшникова", "яйця курячі", "сіль", "перець чорний"],
    "juice": ["яблука", "персики", "вода питна", "цукор", "лимонна кислота", "аскорбінова кислота"],
}
DRUGS = ["Парацетамол", "Ібупрофен", "Цитрамон", "Лоратадин", "Омепразол", "Амоксицилін",
         "Метформін", "Валідол", "Корвалол", "Мукалтин", "Аспірин", "Діклофенак"]
DRUG_SUFFIX = ["ін", "ол", "екс", "фен", "азол", "тин"]
HOUSEHOLD = ["Засіб для миття посуду", "Пральний порошок", "Відбілювач", "Засіб для чищення плит",
             "Рідке мило", "Засіб для миття вікон"]
RECEIPT_ITEMS = ["Хліб житній", "Молоко 2,5%", "Яйця курячі 10 шт", "Сир твердий", "Банани",
                 "Яблука", "Гречка", "Олія соняшникова", "Кава мелена", "Чай чорний", "Вода мінеральна",
                 "Печиво", "Цукор", "Масло вершкове", "Картопля", "Помідори"]
SERVICES = ["електроенергія", "природний газ", "водопостачання", "централізоване опалення",
            "вивезення побутових відходів"]


class Gen:
    def __init__(self, rng: random.Random, lexicon: set):
        self.r = rng
        self.lex = lexicon  # lower-case Ukrainian word list, used to keep nonce/lure words out of it

    # ---------- primitive slot fillers ----------
    def nonce(self, syl=None):
        r = self.r
        for _ in range(200):
            n = syl or r.choice([2, 3, 3])
            w = ""
            for i in range(n):
                on = r.choice(CONS) if r.random() < 0.85 else r.choice(["бр", "гр", "дв", "кл", "пр", "ст", "тр", "зв", "св", "кр"])
                w += on + r.choice(VOWELS)
            if r.random() < 0.5:
                w += r.choice("нрлксмй")
            w = w.replace("ьь", "ь")
            if w not in self.lex and len(w) >= 4:
                return w.capitalize()
        raise RuntimeError("nonce generation failed")

    def lure(self, word):
        """One-letter substitution of a real word (same vowel/consonant class), not in lexicon."""
        r = self.r
        low = word.lower()
        idx = [i for i, c in enumerate(low) if 0 < i < len(low) - 0 and c in VOWELS + CONS]
        for _ in range(200):
            i = r.choice(idx)
            pool = VOWELS if low[i] in VOWELS else CONS
            c = r.choice([p for p in pool if p != low[i]])
            cand = low[:i] + c + low[i + 1:]
            if cand not in self.lex:
                return cand.capitalize() if word[0].isupper() else cand
        raise RuntimeError("lure generation failed")

    def name_slot(self, cond, pool=BRAND_WORDS):
        """A brand/shop/company name in the given condition: word | nonce | lure."""
        if cond == "word":
            return (self.r.choice(pool), "word")
        if cond == "nonce":
            return (self.nonce(), "nonce")
        return (self.lure(self.r.choice(pool)), "lure")

    def surname_slot(self, cond, female=None):
        r = self.r
        if cond == "word":
            return (r.choice(SURNAMES), "name")
        if cond == "nonce":
            return (self.nonce(2) + r.choice(SURNAME_SUFFIX), "nonce")
        return (self.lure(r.choice(SURNAMES)), "lure")

    def date(self):
        r = self.r
        return f"{r.randint(1, 28):02d}.{r.randint(1, 12):02d}.{r.choice([2026, 2027, 2028])}"

    def time(self):
        r = self.r
        return f"{r.randint(6, 22):02d}:{r.choice(['00', '15', '30', '45', '10', '40'])}"

    def hours(self):
        r = self.r
        return f"{r.randint(7, 10):02d}:{r.choice(['00', '30'])}–{r.randint(17, 22)}:{r.choice(['00', '30'])}"

    def phone(self):
        r = self.r
        op = r.choice(["67", "50", "63", "93", "97", "66", "68", "73"])
        return f"+380 {op} {r.randint(100, 999)} {r.randint(10, 99)} {r.randint(10, 99)}"

    def money(self, lo=10, hi=3000):
        v = self.r.randint(lo * 100, hi * 100)
        return f"{v // 100},{v % 100:02d}"

    def fill(self, template):
        out = []
        for t in template:
            if isinstance(t, str):
                out.append((t, "word"))
            elif t[0] == "date":
                out.append((self.date(), "num"))
            elif t[0] == "time":
                out.append((self.time(), "num"))
            elif t[0] == "int":
                out.append((str(self.r.randint(t[1], t[2])), "num"))
        return out

    def person(self, cond):
        r = self.r
        fem = r.random() < 0.5
        sn, cat = self.surname_slot(cond)
        if fem and cat == "name" and sn.endswith(("ак", "ник", "ій")):
            sn = r.choice([s for s in SURNAMES if s.endswith(("енко", "чук", "юк", "ів", "ько"))])
        first = r.choice(FIRST_F if fem else FIRST_M)
        pat = r.choice(PATRON)[0 if fem else 1]
        return [(sn, cat), (" ", "word"), (first, "name"), (" ", "word"), (pat, "name")]

    def address(self):
        r = self.r
        return [("вул. ", "word"), (r.choice(STREETS), "name"), (f", {r.randint(1, 180)}", "num")]

    # ---------- stimulus types ----------
    def signage(self, cond):
        r = self.r
        kind = r.choice(["shop", "shop", "street", "notice", "transport"])
        L = []
        if kind == "shop":
            L.append(("large", [(r.choice(SHOP_TYPES), "word")]))
            nm = self.name_slot(cond)
            L.append(("title", [("«", "word"), nm, ("»", "word")]))
            L.append(("body", [("Пн–Пт ", "word"), (self.hours(), "num")]))
            L.append(("body", [("Сб–Нд ", "word"), (self.hours(), "num")]))
            if r.random() < 0.6:
                L.append(("small", [("тел. ", "word"), (self.phone(), "num")]))
        elif kind == "street":
            nm = self.surname_slot(cond) if r.random() < 0.5 else self.name_slot(cond)
            L.append(("small", [(r.choice(["Галицький", "Шевченківський", "Франківський", "Сихівський", "Залізничний"]), "word"), (" район", "word")]))
            L.append(("title", [("вул. ", "word"), nm]))
            L.append(("large", [(str(r.randint(1, 240)), "num")]))
            if r.random() < 0.5:
                L.append(("small", [("Поштовий індекс ", "word"), (str(r.randint(79000, 79099)), "num")]))
        elif kind == "notice":
            L.append(("title", [(r.choice(["УВАГА!", "ОГОЛОШЕННЯ", "До відома відвідувачів"]), "word")]))
            for tpl in r.sample(NOTICES, 2):
                L.append(("body", self.fill(tpl)))
            L.append(("small", [("Адміністрація ", "word"), ("«", "word"), self.name_slot(cond), ("»", "word")]))
        else:  # transport stop
            L.append(("small", [("Зупинка громадського транспорту", "word")]))
            L.append(("title", [("«", "word"), self.name_slot(cond), ("»", "word")]))
            L.append(("body", [(r.choice(["Трамвай № ", "Тролейбус № ", "Автобус № "]), "word"),
                               (", ".join(str(x) for x in sorted(r.sample(range(1, 70), 3))), "num")]))
            L.append(("body", [("Інтервал руху ", "word"), (str(r.randint(5, 25)), "num"), (" хв", "word")]))
            L.append(("small", [("Перший рейс ", "word"), (f"0{r.randint(5, 6)}:{r.randint(0, 59):02d}", "num"), (", останній ", "word"), (f"2{r.randint(2, 3)}:{r.randint(0, 59):02d}", "num")]))
        return kind, L

    def packaging(self, cond):
        r = self.r
        kind = r.choice(["food", "food", "medicine", "household"])
        L = []
        if kind == "food":
            prod, unit, pool = r.choice(FOODS)
            L.append(("title", [("«", "word"), self.name_slot(cond), ("»", "word")]))
            L.append(("large", [(prod, "word")]))
            if unit == "l":
                L.append(("body", [("Об'єм ", "word"), (r.choice(["0,5", "0,9", "1", "1,5", "0,25"]), "num"), (" л", "word")]))
            else:
                L.append(("body", [("Маса нетто ", "word"), (str(r.choice([90, 180, 200, 250, 300, 350, 400, 450, 500, 800, 900])), "num"), (" г", "word")]))
            ing = r.sample(INGR[pool], min(len(INGR[pool]), r.randint(3, 5)))
            L.append(("small", [("Склад: ", "word"), (", ".join(ing) + ".", "word")]))
            L.append(("body", [(r.choice(["Вжити до: ", "Придатний до: ", "Термін придатності до: "]), "word"), (self.date(), "num")]))
            if r.random() < 0.5:
                lo = r.randint(0, 4); L.append(("small", [("Зберігати при температурі від +", "word"), (f"{lo} до +{lo + r.randint(2, 6)}", "num"), (" °C", "word")]))
            L.append(("small", [("Виробник: ТОВ ", "word"), ("«", "word"), self.name_slot(r.choice(["word", "nonce"])), ("», м. ", "word"), (r.choice(CITIES), "name")]))
        elif kind == "medicine":
            if cond == "word":
                nm = (r.choice(DRUGS), "name")
            elif cond == "nonce":
                nm = (self.nonce(2) + r.choice(DRUG_SUFFIX), "nonce")
            else:
                nm = (self.lure(r.choice(DRUGS)), "lure")
            L.append(("title", [nm]))
            L.append(("large", [(str(r.choice([5, 10, 20, 25, 40, 50, 100, 200, 250, 400, 500])), "num"), (" мг", "word")]))
            L.append(("body", [(str(r.choice([10, 12, 20, 24, 30, 50])), "num"), (" таблеток", "word")]))
            L.append(("small", [("По 1 таблетці ", "word"), (str(r.randint(1, 4)), "num"), (" рази на добу", "word")]))
            L.append(("body", [("Придатний до: ", "word"), (f"{r.randint(1, 12):02d}/{r.choice([2027, 2028, 2029])}", "num")]))
            L.append(("small", [("Серія ", "word"), (str(r.randint(100000, 999999)), "num")]))
            L.append(("small", [("Зберігати в недоступному для дітей місці", "word")]))
        else:
            L.append(("title", [("«", "word"), self.name_slot(cond), ("»", "word")]))
            L.append(("large", [(r.choice(HOUSEHOLD), "word")]))
            L.append(("body", [("Об'єм ", "word"), (str(r.choice([250, 450, 500, 750, 1000])), "num"), (" мл", "word")]))
            L.append(("body", [(r.choice(["Небезпечно! Не ковтати.", "Берегти від дітей!", "Уникати потрапляння в очі."]), "word")]))
            L.append(("small", [("Дата виготовлення: ", "word"), (self.date(), "num")]))
            L.append(("small", [("Гаряча лінія: ", "word"), (self.phone(), "num")]))
        return kind, L

    def document(self, cond):
        r = self.r
        kind = r.choice(["bill", "receipt", "referral", "announcement"])
        L = []
        if kind == "bill":
            L.append(("title", [("РАХУНОК-ПОВІДОМЛЕННЯ", "word")]))
            L.append(("body", [("за ", "word"), (r.choice(MONTHS), "word"), (f" {r.choice([2026, 2027])}", "num"), (" р.", "word")]))
            L.append(("body", [("Послуга: ", "word"), (r.choice(SERVICES), "word")]))
            L.append(("body", [("Особовий рахунок: ", "word"), (str(r.randint(1000000, 9999999)), "num")]))
            L.append(("body", [("Платник: ", "word")] + self.person(cond)))
            L.append(("small", [("Адреса: м. ", "word"), (r.choice(CITIES), "name"), (", ", "word")] + self.address() + [(f", кв. {r.randint(1, 120)}", "num")]))
            L.append(("large", [("Сума до сплати: ", "word"), (self.money(), "num"), (" грн", "word")]))
            L.append(("body", [("Сплатити до ", "word"), (self.date(), "num")]))
        elif kind == "receipt":
            L.append(("large", [(r.choice(["Продукти", "Супермаркет", "Мінімаркет"]), "word"), (" «", "word"), self.name_slot(cond), ("»", "word")]))
            L.append(("small", [("м. ", "word"), (r.choice(CITIES), "name"), (", ", "word")] + self.address()))
            L.append(("body", [("ФІСКАЛЬНИЙ ЧЕК № ", "word"), (str(r.randint(1000, 99999)), "num")]))
            tot = 0
            for it in r.sample(RECEIPT_ITEMS, r.randint(3, 5)):
                v = r.randint(1500, 25000); tot += v
                L.append(("body", [(it + "  ", "word"), (f"{v // 100},{v % 100:02d}", "num")]))
            L.append(("large", [("СУМА  ", "word"), (f"{tot // 100},{tot % 100:02d}", "num")]))
            L.append(("small", [(f"{self.date()} {self.time()}", "num")]))
        elif kind == "referral":
            L.append(("title", [("НАПРАВЛЕННЯ", "word")]))
            L.append(("body", [("Пацієнт: ", "word")] + self.person(cond)))
            L.append(("body", [("Лікар: ", "word"), (r.choice(["терапевт", "кардіолог", "офтальмолог", "невролог", "ендокринолог"]), "word"), (" ", "word"), (r.choice(SURNAMES), "name")]))
            L.append(("large", [("Кабінет № ", "word"), (str(r.randint(100, 450)), "num")]))
            L.append(("body", [("Дата прийому: ", "word"), (f"{self.date()}, {self.time()}", "num")]))
            L.append(("small", [("Мати при собі паспорт та медичну картку", "word")]))
        else:
            L.append(("title", [("ОГОЛОШЕННЯ", "word")]))
            L.append(("body", [("Шановні мешканці!", "word")]))
            L.append(("body", [(self.date(), "num"), (" з ", "word"), (self.time(), "num"), (" до ", "word"), (self.time(), "num")]))
            L.append(("body", [("буде відключено ", "word"), (r.choice(["електроенергію", "холодну воду", "гарячу воду", "газопостачання"]), "word")]))
            L.append(("body", [("у зв'язку з ремонтними роботами.", "word")]))
            L.append(("small", [("Довідки за телефоном ", "word"), (self.phone(), "num")]))
            L.append(("small", [("Правління ОСББ ", "word"), ("«", "word"), self.name_slot(cond), ("»", "word")]))
        return kind, L
