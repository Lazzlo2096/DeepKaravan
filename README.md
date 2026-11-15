  # DeepKaravan

Самостоятельный сервис инференса **BAAI/bge-m3**: FastAPI, Pydantic,
FlagEmbedding, Celery и Redis. Все HTTP-запросы используют одну FIFO-очередь.
Один Celery-воркер с `solo`-пулом загружает одну модель и выполняет задачи последовательно.

## Совместимость с OpenRouter

Karavan должен предоставлять API, совместимый с OpenRouter для поддерживаемых
методов: форматы запросов и ответов, названия полей, HTTP-коды и структура ошибок
должны соответствовать API OpenRouter. Расширения Karavan и любые отклонения
должны быть явно задокументированы в этом файле.

Это требование задаёт направление разработки и не означает, что все методы
OpenRouter уже реализованы. Требования описаны в [SPEC.md](SPEC.md),
инструкции для разработки — в [AGENTS.md](AGENTS.md).

## Этот компьютер: Windows + Radeon 880M

Все три сервиса работают в Docker Desktop: Redis, API и `celery-amd-wsl`.
Celery использует Radeon 880M через WSL `/dev/dxg`, ROCm 7.2.1 и ROCDXG 1.2.2.
Нужны Docker Desktop с WSL2 и совместимый драйвер AMD. Локальный Python
для запуска сервиса не требуется.

В `.env`:

```dotenv
INFERENCE_DEVICE=amd-wsl
COMPOSE_PROFILES=${INFERENCE_DEVICE}
USE_FP16=true
INFERENCE_BATCH_SIZE=1
```

```powershell
just up                   # собрать и запустить три контейнера
just ps                   # состояние контейнеров
just gpu-check            # проверить вычисление на GPU внутри контейнера
just logs-gpu             # загрузка модели и выполнение задач
just restart celery-amd-wsl
just down                 # остановить всё, сохранив данные
```

Можно использовать напрямую `docker compose up -d --build`.
Модель готова после сообщения `BGE-M3 is ready` в логах Celery.
Кэш модели хранится в Docker volume `model-cache`; Redis не публикует порт на хост.
Нативный Windows-воркер больше не запускается. Старые `.venv-amd` и `.cache/huggingface`
не используются и могут быть удалены вручную, если не нужны.

Образ `celery/Dockerfile.wsl` использует официальный PyTorch-образ AMD;
Torch и Triton устанавливаются в нём производителем. Остальные Python-зависимости
зафиксированы в `requirements/amd-wsl.lock`. Для обновления:

```powershell
uv pip compile --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --excludes requirements/amd-wsl-provided.txt requirements/amd-wsl.in -o requirements/amd-wsl.lock
```

Документация: [AMD ROCm / WSL](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html),
[ROCDXG](https://github.com/ROCm/librocdxg).

## Запуск на CPU

Нужен Docker с Compose v2 и доступ в интернет для первой загрузки зависимостей и модели.
CPU- и Linux AMD-образы устанавливают зависимости через uv из общего `uv.lock`.
WSL-образ использует отдельный lock-файл, описанный выше.

```powershell
Copy-Item .env.example .env
docker compose up -d --build
docker compose logs -f celery-cpu
```

В Linux/macOS вместо `Copy-Item` используйте `cp .env.example .env`.
Если `.env` уже существует, редактируйте его, не перезаписывая свои настройки.

API: [Swagger UI](http://localhost:8000/docs),
[OpenAPI](http://localhost:8000/openapi.json).
Первый старт скачивает модель размером примерно 2.3 ГБ; сообщение
`BGE-M3 is ready` в логах воркера означает, что модель загружена.
API может принимать задачи, пока модель ещё загружается.
Кэш модели сохраняется в Docker volume `model-cache`.

```powershell
docker compose ps
docker compose logs -f api celery-cpu
docker compose down
```

`down` сохраняет данные. **`docker compose down -v` удаляет очередь, результаты и кэш модели.**
Порт Redis не публикуется на хост. API по умолчанию слушает на `127.0.0.1:8000`.
Для доступа с других машин задайте `API_HOST=0.0.0.0`; авторизации пока нет.

## Команды just

В проекте есть `justfile` для Windows PowerShell и Linux/macOS.
Нужен [just](https://just.systems/man/en/); для локальных Python-проверок — также `uv`.
Список команд: `just` или `just --list`.

```powershell
just up                  # создать .env при отсутствии, собрать и запустить сервисы
just ps                  # состояние контейнеров
just logs api celery-cpu # логи выбранных сервисов; just logs — все логи
just restart api         # перезапустить API
just down                # остановить сервисы, сохранив данные

just setup               # подготовить .venv с Python 3.12 и зависимости разработки
just check               # lock-файл, Ruff, форматирование, тесты и проверка Compose
just test                # только тесты
just format              # отформатировать Python-код и justfile
just smoke               # проверить запущенный сервис с настоящей моделью
```

`just init` сохраняет существующий `.env`. Выбор CPU/AMD остаётся в `.env`.
Для другого адреса API: `just smoke http://localhost:18091`.
`just recovery` запускает проверку восстановления на CPU: она останавливает воркер
и перезапускает Redis, поэтому запускайте её отдельно от клиентской нагрузки.

## Зависимости и uv

Используется Python 3.12, версия указана в `.python-version`.
Зависимости CPU, API и native Linux AMD объявлены в `pyproject.toml`, а их версии и источники
зафиксированы в `uv.lock`. Оба файла входят в репозиторий.

| Набор | Назначение | Установка |
| --- | --- | --- |
| Общие зависимости | Схемы, настройки, Celery, Redis, токенизатор | Устанавливаются всегда |
| `api` | FastAPI и Uvicorn | `uv sync --locked --no-dev --extra api` |
| `cpu` | FlagEmbedding и CPU-сборка PyTorch | `uv sync --locked --no-dev --extra cpu` |
| `amd` | FlagEmbedding, ROCm-сборка PyTorch и Triton; Linux x86_64 | `uv sync --locked --no-dev --extra amd` |
| `dev` | pytest, Ruff и тестовые зависимости | По умолчанию включён; `--no-dev` исключает его |

Для разработки API и запуска всех тестов достаточно:

```powershell
uv sync --locked --extra api
uv run --locked --extra api pytest -q
```

Этот набор не устанавливает PyTorch и не скачивает веса модели.
Взаимоисключающие extras `cpu` и `amd` привязаны к разным индексам PyTorch;
включать их одновременно нельзя. Команда `uv sync` приводит `.venv` в соответствие
выбранному набору, поэтому для модели и API используйте отдельные окружения/контейнеры.

Локально и при Docker-сборке применяется `--locked`: изменение зависимостей требует
обновить `uv.lock`. Для этого отредактируйте `pyproject.toml` и выполните `uv lock`
(или `just lock`). `just lock-check` проверяет актуальность lock-файла.
Docker-образы используют uv 0.11.29; для локальной разработки нужна эта версия или новее.

## REST API

Базовый URL API: `http://localhost:8000/api/v1`.
Запрос эмбеддингов: `POST http://localhost:8000/api/v1/embeddings`.
Маршруты перенесены с `/v1/*` на `/api/v1/*`; клиентам нужно обновить URL.

Одинаковый запрос для обоих режимов:

```json
{
  "model": "baai/bge-m3",
  "input": ["Первый текст", "Второй текст"],
  "embedding_types": ["dense", "sparse", "colbert"]
}
```

`model` и `input` обязательны в обоих режимах, как в
[контракте OpenRouter](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request).
`model` принимает только `"baai/bge-m3"` без учёта регистра: например,
`"BAAI/BGE-M3"` тоже допустимо. При передаче в очередь значение приводится
к `"baai/bge-m3"`. Это поведение Karavan; регистронезависимость именно
`POST /embeddings` OpenRouter пока не подтверждена.
Отсутствие обязательного поля или другое значение модели возвращает `422`
до постановки задачи в очередь.
`embedding_types` можно сократить до любой непустой комбинации без повторов.
Если `embedding_types` не передано, возвращается только `dense`. `input` содержит от 0 до 1024 строк;
пустой батч допустим и возвращает пустой `data`. Пустая строка тоже допустима.
Ответ сохраняет порядок текстов, `data[i].index` соответствует индексу во входном батче.

**Текущие расширения и отклонения от OpenRouter:**

- `embedding_types`, асинхронный метод и `/api/v1/tasks/*` — расширения Karavan.
- Поддерживается только массив строк в `input`, а не все форматы OpenRouter.
- `data[].embedding` содержит dense-вектор, `data[].object` равен `"embedding"`,
  как в OpenRouter. Поля `sparse`, `colbert`, `task_id` и `embedding_types` —
  расширения Karavan. Если `dense` не запрошен, `embedding` отсутствует —
  это тоже расширение. Поля верхнего уровня `object` и `usage` пока не реализованы.
- Валидация пока возвращает FastAPI `422` с `detail`, тогда как OpenRouter
  документирует `400` с `error.code` и `error.message` для неверных параметров
  ([ошибки OpenRouter](https://openrouter.ai/docs/api/reference/errors-and-debugging)).

Полная совместимость не заявляется.

| Метод и путь | Поведение |
| --- | --- |
| `POST /api/v1/embeddings` | Держать HTTP-соединение до завершения задачи; `200` с результатом или ошибка |
| `POST /api/v1/embeddings/async` | Поставить в общую очередь; сразу `202` с `task_id` |
| `GET /api/v1/tasks/{task_id}` | `200` со статусом и временем создания/завершения |
| `GET /api/v1/tasks/{task_id}/result` | `200` с результатом; `202`, если задача ещё выполняется |
| `GET /health/live` | HTTP-процесс работает |
| `GET /health/ready` | API и токенизатор готовы, Redis доступен; готовность GPU не проверяется |

Состояния: `queued` → `running` → `succeeded` или `failed`.
После прерывания `running` может повторно выполняться с тем же `task_id`.

Ответ `202` содержит `task_id`, `status`, `status_url`, `result_url` и временные поля.
В заголовках — `Location` со ссылкой на статус и `Retry-After: 1`.
Опрос статуса не загружает из Redis сами эмбеддинги.
После истечения срока хранения статус и результат возвращают `404`.
Неизвестный идентификатор тоже даёт `404`, некорректный UUID — `422`.

Результат `200` содержит `task_id`, `model`, `embedding_types`, `data`:

- `embedding`: нормализованный dense-вектор из 1024 чисел (для типа `dense`);
- `sparse`: объект `{ "token_id": weight }`, ключи — строковые ID токенов;
- `colbert`: массив векторов токенов в формате FlagEmbedding, без CLS и padding.

Каждый элемент `data` также содержит `object: "embedding"` и `index`.
Поле `data[].dense` переименовано в `data[].embedding`; клиентам нужно обновить
чтение ответа. Название типа в `embedding_types` остаётся `"dense"`.
Невыбранные типы отсутствуют в ответе. Ошибка инференса отражается в статусе
(`error.code`, `error.message`); получение такого результата возвращает `422`.
Внутренние подробности доступны в логах воркера. При недоступности Redis/очереди API отвечает `503`.

### Пример в PowerShell

```powershell
$body = @{
    model = 'baai/bge-m3'
    input = @('Привет, мир', 'Поиск по документам')
    embedding_types = @('dense', 'sparse', 'colbert')
} | ConvertTo-Json

$task = Invoke-RestMethod `
    -Uri http://localhost:8000/api/v1/embeddings/async `
    -Method Post -ContentType 'application/json; charset=utf-8' `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body))

Invoke-RestMethod "http://localhost:8000$($task.status_url)"
Invoke-RestMethod "http://localhost:8000$($task.result_url)"
```

Для синхронного режима отправьте тот же запрос на `/api/v1/embeddings`.
API держит HTTP-соединение всё время ожидания в очереди и выполнения инференса,
без ограничения длительности запроса на стороне приложения. Ответ `202` с `task_id`
вместо результата в этом режиме не возвращается. При ошибке задачи возвращается ошибка.
Устаревшая настройка `SYNC_TIMEOUT_SECONDS` больше не используется.
Таймауты HTTP-клиента и промежуточного прокси настраиваются отдельно:
для долгих запросов отключите их или задайте достаточное время ожидания.
Разрыв клиентского соединения не отменяет задачу.

### Ограничение длины

Лимит по умолчанию — **8192 токена вместе со специальными токенами**.
API считает длину оригинальным токенизатором зафиксированной ревизии BGE-M3,
не загружая PyTorch или веса модели. Воркер дополнительно проверяет длину.
Если хотя бы один текст превышает лимит, весь запрос отклоняется **до очереди** с `422`,
`code=text_too_long` и индексами слишком длинных текстов. Обрезания и разбиения нет.

`INFERENCE_BATCH_SIZE` задаёт внутренний размер порции модели, а не число текстов
в HTTP-запросе. Для CPU и длинных текстов его можно уменьшить до 1.
Большие результаты ColBERT требуют много RAM и места в Redis; нагрузочные лимиты — в TODO.

## Очередь, результаты и восстановление

- Redis DB 0: брокер Celery, одна очередь `embeddings`, без приоритетов.
- Redis DB 1: статусы и результаты. Общий `JobStore` сохраняет их напрямую,
  `task_ignore_result=True` исключает вторую копию больших результатов в Celery backend.
- Пока задача ожидает или выполняется, её статус не истекает.
- Результат и статус успешной/ошибочной задачи получают TTL **86400 секунд с момента завершения**.
- Redis пишет AOF с `appendfsync always` в volume `redis-data`; очередь восстанавливается
  при перезапуске контейнера с сохранённым volume. Включён `noeviction`.
- `task_acks_late=True`: сообщение подтверждается после сохранения результата или ошибки.
  Прерывание процесса приводит к повторной доставке. Если результат уже сохранён,
  повторная доставка не пересчитывает модель и не продлевает TTL.
- Обычные ошибки инференса завершают задачу состоянием `failed`.
  Прерванная задача и ошибка сохранения в Redis допускают повторное выполнение.

После штатного выключения воркер пытается вернуть неподтверждённое сообщение в очередь.
После `SIGKILL`/сбоя питания возврат может ждать `VISIBILITY_TIMEOUT_SECONDS`
(по умолчанию час). Это таймаут брокера; он не ограничивает длительность HTTP-запроса.
Он должен превышать длительность самого долгого инференса.
Гарантия доставки — **at least once**; абсолютной гарантии единственного вычисления при сбоях нет.
В обычной работе порядок FIFO, но повторная доставка после аварии может изменить порядок.

Данные переживают перезапуск благодаря тому, что volume остаётся на диске.
Один экземпляр Redis не защищает от потери диска/volume и не является резервной копией.

## AMD GPU

Подготовлен отдельный Docker-профиль для Linux-хоста с поддерживаемой AMD GPU,
драйвером ROCm и устройствами `/dev/kfd`, `/dev/dri`.
Без конкретной модели GPU и ОС совместимость оборудования пока не подтверждена.

Перед переключением остановите текущий профиль командой `docker compose down`,
затем измените `.env`:

```dotenv
INFERENCE_DEVICE=amd
COMPOSE_PROFILES=${INFERENCE_DEVICE}
USE_FP16=true
```

```powershell
docker compose up -d --build
docker compose logs -f celery-amd
```

Compose подставляет `INFERENCE_DEVICE` в `COMPOSE_PROFILES` и запускает только один
из `celery-cpu` / `celery-amd` / `celery-amd-wsl`. Включайте один профиль и не масштабируйте воркер
в первой версии. Группы `AMD_VIDEO_GID` и `AMD_RENDER_GID` должны соответствовать хосту.
По умолчанию используется PyTorch 2.8.0 / ROCm 6.4; другие версии потребуют согласовать
образ, PyTorch и драйвер с конкретным железом. Версии и URL индексов задаются в
`pyproject.toml`, затем фиксируются командой `uv lock`; после изменения пересоберите
образы. Индекс не переопределяется через `.env`, чтобы сборка соответствовала lock-файлу.

ROCm использует API `torch.cuda`, поэтому устройство внутри кода — `cuda:0`.
Выбор `amd` проверяет наличие именно ROCm и доступной GPU; тихого перехода на CPU нет.
При `cpu` всегда используется CPU и FP32 независимо от доступных GPU.

## Структура

```text
api/                  FastAPI, публикация задач, Dockerfile
celery/worker/        загрузка модели, инференс, обработка Celery-задач
celery/Dockerfile     отдельные CPU / AMD стадии сборки
redis/redis.conf      AOF, сохранение данных, noeviction
common/               настройки, схемы, токенизатор, Celery-конфигурация, Redis-хранилище
tests/                API, TTL, восстановление выполнения, выбор устройства
scripts/smoke.py      проверка запущенного сервиса с настоящей моделью
scripts/check_recovery.py  проверка сохранения очереди и восстановления после сбоев
compose.yaml          Redis + API + один выбранный воркер
justfile              команды запуска, логов и проверок
pyproject.toml        зависимости, extras, настройки проверок и источники PyTorch
uv.lock               зафиксированные версии и хеши зависимостей
.python-version       версия Python для uv
.env.example          настройки и комментарии
TODO.md               авторизация, webhook и дальнейшая работа
```

У папки `celery/` намеренно нет `__init__.py`, чтобы не перекрыть установленную
библиотеку `celery`. При локальном запуске воркера добавьте `celery/` в `PYTHONPATH`.

## Проверки

Тесты API и обработки задач не скачивают модель и не требуют GPU:

```powershell
uv sync --locked --extra api
uv run --locked --extra api pytest -q
uv run --locked --extra api ruff check .
uv run --locked --extra api ruff format --check .
uv lock --check
docker compose config --quiet
```

Проверка запущенного сервиса с настоящей моделью:

```powershell
uv run --locked --extra api python scripts/smoke.py --base-url http://localhost:8000
```

Проверка восстановления с настоящей моделью (запускайте отдельно от клиентской нагрузки:
она останавливает CPU-воркер и перезапускает Redis):

```powershell
uv run --locked --extra api python scripts/check_recovery.py --base-url http://localhost:8000
```

Она проверяет сохранение очереди, FIFO, TTL
и повторный инференс после принудительного завершения воркера.

Документация используемых компонентов:
[BGE-M3](https://bge-model.com/bge/bge_m3.html),
[Celery / Redis](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html),
[Redis persistence](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/),
[uv / Docker](https://docs.astral.sh/uv/guides/integration/docker/),
[uv / PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/),
[PyTorch CPU / ROCm](https://pytorch.org/get-started/previous-versions/).
