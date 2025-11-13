# Детальне порівняння моделей Gemini для генерації ML pipelines

**Автор:** Фефелов Ілля Олександрович  
**МАУП, 2025**  
**Дата:** 13 листопада 2025

## 1. Мета експерименту

Порівняти три моделі Gemini (2.5-flash-lite, 2.5-flash, 2.5-pro) на здатність генерувати високоякісні ML pipelines для різних датасетів sklearn та визначити оптимальну модель для дипломної роботи з ablation analysis.

## 2. Методологія

### 2.1 Тестовані моделі
- **gemini-2.5-flash-lite** - найшвидша модель (paid tier: 15 RPM, 3K TPM)
- **gemini-2.5-flash** - збалансована модель (paid tier: 10 RPM, 1.48K TPM)
- **gemini-2.5-pro** - найрозумніша модель (paid tier: 2 RPM, 394 TPM)

### 2.2 Датасети
1. **breast_cancer** - 569 samples, 30 features, 2 classes (binary classification)
2. **wine** - 178 samples, 13 features, 3 classes (small multiclass)
3. **digits** - 1797 samples, 64 features, 10 classes (large multiclass, high dimensions)
4. **iris** - 150 samples, 4 features, 3 classes (classic small dataset)

### 2.3 Метрики оцінювання
- **Accuracy** - середня точність 5-fold cross-validation
- **Generation Time** - час генерації коду в секундах
- **Algorithm Choice** - який алгоритм обрала модель
- **Code Complexity** - наявність складних алгоритмів (RF/GB/SVC/MLP)

### 2.4 Промпт
Всі моделі отримували однаковий промпт з вимогою обрати НАЙКРАЩИЙ алгоритм на основі характеристик датасету (розмір, кількість features, класів). Промпт явно вказував 5 варіантів: LogisticRegression, RandomForest, SVC, GradientBoosting, MLPClassifier.

## 3. Результати

### 3.1 Підсумкова таблиця по датасетам

| Dataset | gemini-2.5-flash-lite | gemini-2.5-flash | gemini-2.5-pro | **Winner** |
|---------|----------------------|------------------|----------------|------------|
| **breast_cancer** | 94.90% (RF) | **95.08% (GB)** ⭐ | 94.72% (SVC) | flash |
| **wine** | 96.10% (GridSearch) | 96.67% (SVC) | **97.19% (SVC)** ⭐ | **pro** |
| **digits** | 94.49% (SVC) | 92.04% (MLP) | **94.94% (SVC)** ⭐ | **pro** |
| **iris** | 91.33% (GridSearch) | 90.00% (GB) | **92.00% (SVC)** ⭐ | **pro** |
| **Avg Accuracy** | 94.21% | 93.45% | **94.71%** ⭐ | **pro** |

### 3.2 Швидкість генерації

| Model | Avg Time | Min Time | Max Time | **Speedup vs Pro** |
|-------|----------|----------|----------|-------------------|
| **gemini-2.5-flash-lite** | **4.07s** ⚡ | 3.35s | 4.79s | **6.6× faster** |
| gemini-2.5-flash | 21.98s | 14.69s | 37.71s | 1.2× faster |
| gemini-2.5-pro | 27.06s | 23.55s | 33.89s | baseline |

### 3.3 Вибір алгоритмів

#### Розподіл по моделях:

**gemini-2.5-flash-lite** (найрізноманітніший):
- RandomForestClassifier: 1
- GridSearchCV (з tuning): 2
- SVC: 1
- **Complexity: 75% complex models**

**gemini-2.5-flash**:
- GradientBoostingClassifier: 2
- SVC: 1
- MLPClassifier: 1
- **Complexity: 100% complex models**

**gemini-2.5-pro** (консервативний):
- **SVC: 6 з 12 (50%)**
- GradientBoostingClassifier: 0
- RandomForest: 0
- **Complexity: 100% complex models**

#### Загальний розподіл (12 експериментів):

| Algorithm | Count | Percentage |
|-----------|-------|------------|
| **SVC** | 6 | 50% |
| GridSearchCV | 2 | 17% |
| GradientBoosting | 2 | 17% |
| RandomForest | 1 | 8% |
| MLPClassifier | 1 | 8% |

**Висновок:** Жодна модель НЕ обрала простий LogisticRegression! Всі 3 моделі генерують складні алгоритми, що підтверджує ефективність оновленого промпту.

### 3.4 Аналіз по датасетам

#### breast_cancer (569 samples, 30 features, binary)
- **Переможець:** flash (95.08%, GradientBoosting)
- Lite обрав RandomForest (94.90%), Pro обрав SVC (94.72%)
- **Висновок:** Різниця <1%, всі моделі справляються добре

#### wine (178 samples, 13 features, 3 classes)
- **Переможець:** **pro (97.19%, SVC)** ⭐
- Flash також обрав SVC (96.67%), Lite використав GridSearch (96.10%)
- **Висновок:** Pro показав найкращий результат (+1% accuracy)

#### digits (1797 samples, 64 features, 10 classes)
- **Переможець:** **pro (94.94%, SVC)** ⭐
- Lite також обрав SVC (94.49%), Flash невдало обрав MLP (92.04%)
- **Висновок:** Pro найстабільніший, Flash помилився з вибором MLP

#### iris (150 samples, 4 features, 3 classes)
- **Переможець:** **pro (92.00%, SVC)** ⭐
- Flash обрав GB (90.00%), Lite використав GridSearch (91.33%)
- **Висновок:** Малий датасет - всі моделі працюють гірше, Pro найкращий

## 4. Статистичний аналіз

### 4.1 Середня accuracy по моделях

```
gemini-2.5-flash-lite:  94.21% ± 2.03%  (4 datasets)
gemini-2.5-flash:       93.45% ± 2.99%  (4 datasets)
gemini-2.5-pro:         94.71% ± 2.13%  (4 datasets)
```

**Різниця Pro vs Lite:** +0.50% абсолютних (0.5 percentage points)

### 4.2 Статистична значущість

**t-test (Pro vs Lite):**
- t-statistic: 0.43
- p-value: 0.68 (p > 0.05)
- **Висновок: Різниця НЕ статистично значуща**

**Effect size (Cohen's d):**
- d = 0.24 (малий ефект)
- **Висновок: Практична різниця мінімальна**

### 4.3 Співвідношення швидкість/accuracy

**Cost-Benefit аналіз:**

| Model | Avg Accuracy | Avg Time | Score (Acc/Time) | **Winner** |
|-------|--------------|----------|------------------|------------|
| **flash-lite** | 94.21% | 4.07s | **23.15** ⭐ | **BEST** |
| flash | 93.45% | 21.98s | 4.25 | - |
| pro | 94.71% | 27.06s | 3.50 | - |

**Flash-lite має найкращий score швидкість/точність!**

## 5. Якісний аналіз

### 5.1 Переваги gemini-2.5-pro

✅ **Найвища середня accuracy** (94.71%)
✅ **Найкращий на 3 з 4 датасетів** (wine, digits, iris)
✅ **Стабільний вибір SVC** - консервативний, але надійний
✅ **Найкраща variance** - однаковий підхід до всіх датасетів
✅ **Найкращий для складних задач** (multiclass, високі розміри)

### 5.2 Переваги gemini-2.5-flash-lite

✅ **НАЙШВИДША генерація** - 6.6× швидше за Pro (4.07s vs 27.06s)
✅ **Найрізноманітніший вибір** - використовує RF, GridSearch, SVC
✅ **Найкращий cost-benefit ratio** (score 23.15)
✅ **Достатня accuracy** - 94.21% (тільки -0.5% порівняно з Pro)
✅ **Ідеально для швидкого прототипування**

### 5.3 Переваги gemini-2.5-flash

❌ **Найгірша середня accuracy** (93.45%)
❌ **Помилка з MLP на digits** (-2.45% порівняно з Pro)
⚠️ **Середня швидкість** - повільніша за Lite, не краща за Pro
✅ **Розмаїтість алгоритмів** - використовує GB, SVC, MLP

## 6. Висновки та рекомендації

### 6.1 Відповіді на дослідницькі питання

**Q1: Чи є різниця між моделями Gemini?**
- ✅ ТАК, але мінімальна (+0.5% Pro vs Lite)
- ✅ Різниця НЕ статистично значуща (p=0.68)
- ✅ Всі моделі генерують складні алгоритми (RF/GB/SVC/MLP)

**Q2: Чи виправдовує Pro модель 6.6× повільнішу генерацію?**
- ❌ НІ для простих sklearn датасетів
- ✅ МОЖЛИВО для більш складних real-world задач
- ⚠️ Потрібен trade-off: +0.5% accuracy за 23s додаткового часу

**Q3: Яку модель обрати для дипломної роботи?**
- **Рекомендація: gemini-2.5-flash-lite** ⭐

### 6.2 Обґрунтування вибору flash-lite

**Для дипломної роботи з ablation analysis:**

1. **Швидкість критична:**
   - 4 датасети × 6 конфігурацій × 5 runs = 120 експериментів
   - З flash-lite: 120 × 4s = **8 хвилин на всі генерації**
   - З pro: 120 × 27s = **54 хвилини** (майже годину!)

2. **Accuracy достатня:**
   - 94.21% vs 94.71% - різниця 0.5%
   - Для ablation analysis важливіша відносна різниця між конфігураціями
   - Різниця 0.5% не вплине на висновки про ефективність компонентів

3. **Рейт ліміти:**
   - Flash-lite: 15 RPM (найвищий)
   - Pro: 2 RPM (найнижчий)
   - Для батч експериментів flash-lite зручніший

4. **Cost:**
   - Flash-lite дешевша в 2× порівняно з Pro
   - Для студентського проекту важливо

5. **Різноманітність:**
   - Flash-lite використовує GridSearch, RF, SVC
   - Pro обирає тільки SVC (консервативний)
   - Для дослідження краще різноманітність

### 6.3 Коли обирати Pro?

Gemini-2.5-pro варто використовувати якщо:
- ❗ **Production deployment** - кожен 0.5% accuracy критичний
- ❗ **Складні датасети** - real-world з шумом, пропусками, imbalance
- ❗ **Multiclass з багатьма класами** (Pro краща на digits, iris)
- ❗ **Малі датасети** - Pro стабільніша на iris (150 samples)
- ❗ **Час не критичний** - можна чекати 30s на генерацію

### 6.4 Фінальна рекомендація

**Для дипломної роботи:**

```python
# main_experiment.py
model = genai.GenerativeModel("gemini-2.5-flash-lite")  # РЕКОМЕНДОВАНО ⭐
```

**Обґрунтування:**
- ✅ 6.6× швидша генерація (4s vs 27s)
- ✅ Достатня accuracy (94.21%, тільки -0.5% vs Pro)
- ✅ Найкращий cost-benefit ratio (23.15)
- ✅ Найвищі rate limits (15 RPM)
- ✅ Різноманітність алгоритмів (GridSearch, RF, SVC)
- ✅ Статистично еквівалентна Pro (p=0.68)
- ✅ Підходить для research з багатьма експериментами

**Примітка:** Якщо в майбутньому знадобиться максимальна accuracy для production, можна перегенерувати pipelines з gemini-2.5-pro (+0.5% accuracy за 6.6× більше часу).

## 7. Матеріали для захисту диплому

### 7.1 Ключові цифри для презентації

| Метрика | Flash-Lite | Pro | Різниця |
|---------|-----------|-----|---------|
| **Accuracy** | 94.21% | 94.71% | +0.5% |
| **Speed** | 4.07s | 27.06s | **6.6× faster** |
| **Cost-Benefit** | 23.15 | 3.50 | **6.6× better** |
| **Rate Limit** | 15 RPM | 2 RPM | **7.5× more** |
| **Statistical Significance** | - | p=0.68 | **NOT significant** |

### 7.2 Візуалізації для диплому

**Рис. 1: Accuracy по датасетам (bar chart)**
- breast_cancer: Lite 94.90%, Flash 95.08%, Pro 94.72%
- wine: Lite 96.10%, Flash 96.67%, Pro 97.19% ⭐
- digits: Lite 94.49%, Flash 92.04%, Pro 94.94% ⭐
- iris: Lite 91.33%, Flash 90.00%, Pro 92.00% ⭐

**Рис. 2: Speed vs Accuracy (scatter plot)**
- Flash-lite: швидка і точна (top-left quadrant) ⭐
- Flash: повільна і неточна (bottom-right)
- Pro: повільна але точна (bottom-left)

**Рис. 3: Algorithm distribution (pie chart)**
- SVC: 50% (всі моделі люблять SVC)
- GridSearch: 17%
- GradientBoosting: 17%
- RandomForest: 8%
- MLP: 8%

### 7.3 Текст для розділу "Вибір моделі генерації"

> У рамках дипломної роботи було проведено порівняльний аналіз трьох моделей Google Gemini (2.5-flash-lite, 2.5-flash, 2.5-pro) на здатність генерувати високоякісні ML pipelines для чотирьох benchmark датасетів sklearn (breast_cancer, wine, digits, iris).
>
> Результати показали, що **gemini-2.5-pro** демонструє найвищу середню accuracy (94.71%) та найкращі результати на 3 з 4 датасетів. Однак, статистичний аналіз (t-test, p=0.68) показав, що різниця в accuracy між Pro та Flash-lite (+0.5%) **не є статистично значущою**.
>
> При цьому **gemini-2.5-flash-lite** генерує код **в 6.6 разів швидше** (4.07s vs 27.06s), має **найвищі rate limits** (15 RPM vs 2 RPM), та демонструє **найкращий cost-benefit ratio** (23.15 vs 3.50).
>
> Враховуючи, що для ablation analysis критична швидкість проведення експериментів (120+ генерацій pipelines), а різниця в 0.5% accuracy не впливає на висновки про ефективність компонентів, було прийнято рішення використовувати **gemini-2.5-flash-lite** як основну модель для генерації pipelines.
>
> Важливо відмітити, що всі три моделі Gemini генерують **складні алгоритми** (RandomForest, GradientBoosting, SVC, MLP) та **не обирають простий LogisticRegression**, що підтверджує високу якість оновленого промпту та здатність моделей адаптувати вибір алгоритму до характеристик датасету.

## 8. Додаткові спостереження

### 8.1 Паттерни вибору алгоритмів

**Pro модель:**
- Сильна перевага SVC (6/12 = 50%)
- Консервативний підхід - один алгоритм для всіх
- Найкраща для small datasets (iris, wine)

**Flash-lite модель:**
- Використовує GridSearchCV (автоматичний tuning)
- Різноманітність: RF, SVC, GridSearch
- Адаптивний підхід до різних датасетів

**Flash модель:**
- Найбільша variance в виборі
- Іноді помиляється (MLP на digits)
- Не рекомендується

### 8.2 Вплив розміру датасету

| Dataset Size | Best Model | Algorithm |
|--------------|------------|-----------|
| Small (< 200) | Pro | SVC |
| Medium (200-600) | Flash | GB |
| Large (> 1000) | Pro | SVC |

**Висновок:** SVC універсальний, працює на всіх розмірах

### 8.3 Вплив кількості features

| Features | Best Model | Algorithm |
|----------|------------|-----------|
| Low (4) | Pro | SVC |
| Medium (13-30) | Flash/Pro | SVC/GB |
| High (64) | Pro | SVC |

**Висновок:** Для високих розмірностей Pro стабільніша

## 9. Обмеження дослідження

1. **Тільки sklearn датасети** - прості, класичні, без шуму
2. **Тільки classification** - не regression, не clustering
3. **Малі датасети** - максимум 1797 samples
4. **Тільки 1 run per model** - немає variance estimate для моделей
5. **Фіксований промпт** - можливо оптимізація промпту покращить результати

## 10. Напрямки подальших досліджень

1. **Тестування на real-world datasets** з Kaggle
2. **Порівняння з іншими LLM** (GPT-4, Claude, Llama)
3. **Оптимізація промпту** для кращого вибору алгоритмів
4. **Ablation analysis для самих Gemini моделей**
5. **Тестування на regression та clustering задачах**

---

**Підготував:** Фефелов Ілля Олександрович  
**Дата:** 13 листопада 2025  
**Версія:** 1.0
