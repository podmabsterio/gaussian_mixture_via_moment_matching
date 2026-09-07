# Gaussian Mixtures via Moment Matching

Исследовательский репозиторий для восстановления параметров конечной смеси
гауссовских распределений по локализованным моментам и для воспроизводимого
сравнения моделей на синтетических данных.

## Идея метода

Пусть наблюдения независимо получены из изотропной гауссовской смеси

\[
f(x)=\sum_{k=1}^{K}\mu_k\,\sigma_k^{-d}
\varphi\!\left(\frac{x-m_k}{\sigma_k}\right).
\]

Задача — восстановить центры \(m_k\), стандартные отклонения \(\sigma_k\) и
веса \(\mu_k\). Вместо максимизации правдоподобия метод сопоставляет
эмпирические локализованные моменты
\(Z_j=n^{-1}\sum_i\omega_j(X_i)\) с их аналитическим откликом для смеси и
минимизирует расхождение между ними. В качестве тестовых функций используются
гауссовские ядра, а также первые, направленные вторые и радиальные вторые
моменты вокруг выбранных центров.

Перепараметризация поглощает множитель, зависящий от размерности, в амплитуды
компонент. Радиальные моменты агрегируют информацию по направлениям и дают
более прямой сигнал для оценки изотропных дисперсий. Постановка и выводы
приведены в [problem_statement.pdf](./problem_statement.pdf).

Репозиторий содержит NumPy/SciPy-реализацию метода, генераторы синтетических
данных, Hydra-пайплайн для серий экспериментов и web-интерфейс. В UI можно
запускать серии, наблюдать одну оптимизацию по итерациям, просматривать
PCA-проекции и сравнивать сохраненные результаты из `results/`.

## Установка и запуск UI

Нужны Git и Python 3.10 или новее (рекомендуется Python 3.11/3.12).

```bash
git clone https://github.com/podmabsterio/gaussian_mixture_via_moment_matching.git
cd gaussian_mixture_via_moment_matching

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m ui.start
```

На Windows окружение активируется командой `.venv\Scripts\activate`.
После запуска откройте <http://127.0.0.1:8080>. Другой адрес можно задать так:

```bash
python -m ui.start --host 0.0.0.0 --port 8080
```

Для разработки установите дополнительные инструменты и запустите тесты:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Основные директории

| Путь | Назначение |
| --- | --- |
| `src_np/` | Реализация моделей, моментных откликов и оптимизаторов |
| `experiments/synthetic_data/` | Генераторы синтетических датасетов |
| `experiments/metrics/` | Метрики и агрегаторы |
| `experiments/configs/` | Hydra-конфиги воспроизводимых серий экспериментов |
| `ui/declarations/` | JSON-описания моделей, датасетов, параметров запуска и метрик |
| `ui/backend/` | Валидация деклараций, сборка Hydra-конфига и запуск задач |
| `ui/frontend/` | Статический frontend |
| `results/` | Сохраненные запуски, которые читают Results и Comparison |

## Как расширять репозиторий

### Добавление модели

Наследование от специального базового класса не требуется. Класс, доступный из
UI, должен соблюдать следующий контракт:

1. Конструктор принимает `n_components`, `random_state` и все параметры,
   объявленные для модели в JSON. UI-раннер сам передает истинное число
   компонент датасета и seed модели.
2. `fit(X, iteration_callback=None, **dataset)` обучает модель и возвращает
   `self`. `**dataset` позволяет не ломаться на дополнительной разметке или
   служебных полях генератора.
3. `params_dict()` возвращает словарь с NumPy-массивами:
   `means` формы `(K, D)`, `weights` формы `(K,)` и `covariances` формы
   `(K, D, D)`. Для сферической смеси разрешена форма `(K,)`, но это должны
   быть **дисперсии**, а не стандартные отклонения.
4. Для Quick Experiment модель вызывает необязательный `iteration_callback`
   после инициализации и после каждой содержательной итерации. Исключения из
   callback нельзя подавлять: через них, в частности, останавливается запуск.

Минимальный каркас:

```python
import numpy as np

from src_np import IterationSnapshot, MixtureParameters


class MyGaussianMixtureModel:
    def __init__(
        self,
        n_components,
        learning_rate=0.01,
        max_steps=50,
        random_state=None,
    ):
        self.n_components = int(n_components)
        self.learning_rate = float(learning_rate)
        self.max_steps = int(max_steps)
        self.random_state = random_state

    def fit(self, X, iteration_callback=None, **dataset):
        X = np.asarray(X, dtype=float)
        self._initialize(X)

        for iteration in range(self.max_steps + 1):
            loss = self._objective(X)
            if iteration_callback is not None:
                iteration_callback(
                    IterationSnapshot(
                        iteration=iteration,
                        loss=loss,
                        parameters=MixtureParameters.spherical(
                            self.means_, self.weights_, self.sigmas_
                        ),
                        phase="initialization" if iteration == 0 else "optimization",
                    )
                )
            if iteration < self.max_steps:
                self._step(X)
        return self

    def params_dict(self):
        return {
            "means": self.means_,
            "weights": self.weights_,
            "covariances": np.square(self.sigmas_),
        }
```

`MixtureParameters.spherical(...)` принимает именно стандартные отклонения и
сам строит полные ковариационные матрицы. Для общей ковариации используйте
`MixtureParameters(means, weights, covariances)`. `iteration`, `loss` и все
массивы должны быть конечными; номера итераций — неотрицательными и
возрастающими. В `losses` можно передать составляющие функции потерь, а в
`metadata` — небольшие диагностические значения.

После реализации добавьте элемент в
`ui/declarations/models_declaration.json`:

```json
{
  "id": "my_gmm",
  "display_name": "My Lovely Model",
  "description": "Краткое описание модели.",
  "target": "my_package.models.MyGaussianMixtureModel",
  "default_instance_name": "my_gmm",
  "parameters": [
    {
      "key": "learning_rate",
      "display_name": "Learning rate",
      "description": "Размер шага оптимизатора.",
      "type": "number",
      "default": 0.01,
      "exclusive_minimum": 0.0,
      "maximum": 1.0,
      "step": 0.01
    }
  ]
}
```

`target` должен быть импортируемым полным путем к классу. После изменения
деклараций перезапустите UI: они загружаются и проверяются при старте.

### Добавление датасета

Генератор должен принимать объявленные параметры в конструкторе и иметь метод
`generate(seed)`. Он возвращает словарь как минимум с полями:

| Поле | Форма | Содержание |
| --- | --- | --- |
| `X` | `(N, D)` | Наблюдения |
| `true_means` | `(K, D)` | Истинные центры |
| `true_weights` | `(K,)` | Истинные веса |
| `true_covariances` | `(K, D, D)` | Истинные ковариации |
| `true_labels` | `(N,)` | Номер компоненты каждого наблюдения |

Дополнительные поля разрешены: весь словарь передается в `model.fit` и
метрики. Например, можно вернуть `evaluation_X`, `evaluation_labels`,
`evaluation_true_log_pdf` или маску загрязнения.

Каркас генератора:

```python
class MyDatasetGenerator:
    def __init__(self, n_features, n_components, n_samples, noise=0.0):
        self.n_features = int(n_features)
        self.n_components = int(n_components)
        self.n_samples = int(n_samples)
        self.noise = float(noise)

    def generate(self, seed):
        # Используйте локальный RNG: np.random.default_rng(seed).
        return {
            "X": X,
            "true_means": means,
            "true_weights": weights,
            "true_covariances": covariances,
            "true_labels": labels,
        }
```

Зарегистрируйте класс в `ui/declarations/datasets_declaration.json`. Повторяемые
параметры удобно вынести в корневой объект `parameter_sets` и подключить к
датасету через `"parameter_sets": ["имя_набора"]`; локальный список
`parameters` добавляется после общего набора.

### JSON-декларации и параметры UI

Во всех трех файлах используется `"schema_version": 1`. У модели и датасета
обязательны `id`, `display_name`, `description`, `target`,
`default_instance_name` и `parameters`. Каждый параметр содержит `key`,
`display_name`, `description`, `type` и `default`.

Поддерживаемые типы:

- `integer`, `number`, `boolean`, `string`, `enum`;
- `integer_array`, `number_array`.

Доступные ограничения:

- для чисел: `minimum`, `maximum`, `exclusive_minimum`, `exclusive_maximum` и
  шаг элемента управления `step`;
- для строк: `maximum_length`;
- для массивов: `min_items`, `max_items` и числовые границы каждого элемента;
- для `enum`: непустой `options` из объектов с `value` и `display_name`;
- `nullable: true` разрешает `null`;
- `conflicts_with: ["other_key"]` запрещает одновременные ненулевые значения.

Два флага управляют видимостью:

- `advanced: true` скрывает поле до включения Advanced, но пользователь может
  его изменить;
- `hidden: true` никогда не показывает поле, всегда подставляет его `default`
  на backend и запрещает переопределение с клиента.

Значение `default` из декларации является значением по умолчанию именно для UI
и может отличаться от значения в сигнатуре Python-класса. Backend передает в
конструктор все разрешенные параметры, включая значения по умолчанию.

`ui/declarations/execution_parameters.json` содержит три независимые части:

- `parameters` — настройки серийного Experiment; `path` указывает место в
  собираемом Hydra-конфиге, а `scope: "manager"` оставляет настройку менеджеру
  UI;
- `quick_parameters` — настройки одиночного Quick Experiment;
- `metrics` — единый список метрик для UI и собранных им конфигов;
- `fixed_config` — неизменяемые runner и агрегаторы.

Новый чисто конфигурационный параметр запуска достаточно связать через `path`,
например `"path": "runner.n_jobs"`. Если параметр меняет логику менеджера или
Quick Experiment, его также нужно обработать в соответствующем backend-коде.

### Добавление метрики

1. Создайте наследника `experiments.metrics.base_metric.BaseMetric`.
2. Передайте короткое уникальное имя в `super().__init__(...)`.
3. Реализуйте `__call__`, верните один конечный `float` и добавьте `**kwargs`,
   чтобы единый evaluator мог передавать весь контекст.
4. Экспортируйте класс из `experiments/metrics/__init__.py` либо используйте
   полный путь к модулю в `target`.
5. Добавьте декларацию метрики в `execution_parameters.json`.

Пример:

```python
import numpy as np

from experiments.metrics.base_metric import BaseMetric


class MeanCenterBias(BaseMetric):
    def __init__(self):
        super().__init__("Mean center bias")

    def __call__(self, means, true_means, **kwargs):
        return float(np.mean(np.linalg.norm(means - true_means, axis=1)))
```

Evaluator предварительно сопоставляет оцененные компоненты с истинными и может
передать метрике `means`, `weights`, `covariances`, `labels`, все поля датасета
и `model`. Декларация выглядит так:

```json
{
  "id": "mean_center_bias",
  "target": "experiments.metrics.MeanCenterBias",
  "display_name": "Mean center bias",
  "description": "Среднее расстояние между сопоставленными центрами.",
  "direction": "minimize",
  "format": ".5g",
  "track_during_fit": true,
  "arguments": {}
}
```

`direction` принимает `minimize`, `maximize` или `neutral`. При
`track_during_fit: false` метрика считается только в конце Quick Experiment;
это полезно для дорогих метрик. `arguments` передаются в конструктор метрики.

### Hydra-конфиги для серийных запусков

JSON-декларации нужны UI. Для версионируемого запуска без UI создайте YAML в
`experiments/configs/`. Минимальная структура:

```yaml
run_name: my_experiment
save_dir: results
override: false
save_data_visualizations: false
num_visualization_samples: 0

runner:
  _target_: experiments.runners.data_parallel_runner.DataParallelRunner
  _recursive_: false
  num_datasets: 5
  model_seeds_per_dataset: 2
  n_jobs: 1
  threads_limit: 1
  save_raw_results: true

models:
  - model_name: my_gmm
    target:
      _target_: my_package.models.MyGaussianMixtureModel
      learning_rate: 0.01

datasets:
  - name: gaussian_d10
    target:
      _target_: experiments.synthetic_data.gaussian.GaussianDatasetGenerator
      n_features: 10
      n_components: 3
      n_samples: 500

metrics:
  - _target_: experiments.metrics.ForwardKL
  - _target_: experiments.metrics.MeanCenterBias

aggregators:
  - _target_: experiments.metrics.aggregators.MeanAggregator
  - _target_: experiments.metrics.aggregators.StdAggregator
```

Имя файла передается явно, без расширения:

```bash
python run.py --config-name my_experiment
```

Проверить итоговый Hydra-конфиг без запуска можно командой
`python run.py --config-name my_experiment --cfg job`. Результат сохраняется в
`results/<run_name>/`: `config.yaml` в корне, а CSV с агрегатами и, если
включено, сырыми повторами — в подпапках датасетов. Этот же формат читает UI.

## Проверка изменения

Перед коммитом достаточно выполнить:

```bash
python -m pytest -q
python -c "from ui.backend.declarations import DeclarationStore; DeclarationStore('ui/declarations')"
```

Вторая команда сразу обнаруживает невалидный JSON, дублирующиеся `id`/`key` и
неимпортируемые `target`.
