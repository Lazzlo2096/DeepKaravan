# DeepKaravan

Сервис эмбеддингов BAAI/bge-m3: FastAPI и отдельный процесс инференса
в **одном Docker-контейнере**, без Redis и Celery. API передаёт задачи через
`multiprocessing.Queue`, воркер возвращает события и результаты через вторую очередь.
Одна модель выполняет задачи последовательно в порядке FIFO.

## Совместимость с OpenRouter

Требования описаны в [SPEC.md](SPEC.md), инструкции разработки — в [AGENTS.md](AGENTS.md).
Контракт сверяется с [официальной документацией OpenRouter](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request).
Полная совместимость не заявляется; расширения и отклонения перечислены ниже.

## Запуск

Нужны Docker и Docker Compose. Для команд `just` также нужен just;
для локальной разработки — uv 0.11.29 или новее и Python 3.12.

```powershell
just init
# Отредактируйте .env под своё устройство.
just up
just logs
```

Без just: скопируйте `.env.example` в `.env`, настройте его и выполните
`docker compose up -d --build`. API слушает `http://localhost:8000`.
Swagger UI: `http://localhost:8000/docs`, OpenAPI: `/openapi.json`.

Один корневой `Dockerfile` содержит стадии `cpu`, `amd` и `amd-wsl`.
Compose собирает стадию из `INFERENCE_DEVICE`; все варианты запускают только сервис `api`.
`python -m api` запускает один Uvicorn-процесс, который создаёт один процесс `gpu_worker`.
Не используйте несколько Uvicorn workers, reload или несколько реплик: у каждого
будут отдельные очереди, хранилище задач и копия модели.

### Windows / Docker Desktop / AMD WSL

Для Radeon 880M и аналогичных поддерживаемых AMD GPU в `.env`:

```dotenv
INFERENCE_DEVICE=amd-wsl
COMPOSE_PATH_SEPARATOR=;
COMPOSE_FILE=compose.yaml;compose.amd-wsl.yaml
INFERENCE_BATCH_SIZE=1
USE_FP16=true
```

Нужны Docker Desktop с WSL2 и подходящий AMD-драйвер на Windows. Используются
`/dev/dxg`, библиотеки `/usr/lib/wsl/lib`, ROCm 7.2.1 и ROCDXG 1.2.2.
Образ AMD и пакет ROCDXG закреплены digest/checksum в Dockerfile.
Torch/Triton берутся из образа; остальные зависимости — из `requirements/amd-wsl.lock`.

```powershell
just gpu-check
just logs-gpu
```

### CPU и AMD Linux

CPU:

```dotenv
INFERENCE_DEVICE=cpu
COMPOSE_PATH_SEPARATOR=;
COMPOSE_FILE=compose.yaml
USE_FP16=false
```

AMD на Linux с ROCm:

```dotenv
INFERENCE_DEVICE=amd
COMPOSE_PATH_SEPARATOR=;
COMPOSE_FILE=compose.yaml;compose.amd.yaml
USE_FP16=true
AMD_VIDEO_GID=44
AMD_RENDER_GID=109
```

Для Linux нужны `/dev/kfd`, `/dev/dri` и корректные группы video/render.
CPU и Linux AMD используют Torch 2.8.0 и отдельные индексы CPU/ROCm 6.4.
Поддержку конкретной GPU необходимо проверять. Автоматического отката AMD → CPU нет.
После изменения устройства или зависимостей выполните `just up` для пересборки.
Кэш модели хранится в volume `model-cache` и сохраняется при `just down`.

## Архитектура и жизненный цикл

```text
один контейнер
  api/       FastAPI + статусы/результаты в памяти + поток приёма результатов
      |      multiprocessing.Queue: (task_id, запрос)
      v
  gpu_worker/  отдельный spawn-процесс, одна модель, последовательный инференс
      |      multiprocessing.Queue: ready / running / succeeded / failed
      v
  api/       HTTP-ответ или результат для опроса по task_id
```

Модель и Torch загружаются только в дочернем процессе. Запуск через `spawn`
используется на Windows и Linux, без наследования GPU-состояния через fork.
`/health/ready` возвращает `200` только после загрузки модели и пока воркер жив.
При ошибке загрузки или превышении startup timeout запуск API завершается ошибкой.

- Одна общая FIFO-очередь для sync и async запросов. `QUEUE_CAPACITY` ограничивает
  количество ожидающих задач; переполнение возвращает `503` без создания задачи.
- Состояния: `queued` → `running` → `succeeded` / `failed`. Ошибка отдельной задачи
  не останавливает очередь. Подробности ошибки находятся в логах воркера.
- Статусы и результаты хранятся **в памяти API**. Для завершённых задач действует
  `RESULT_TTL_SECONDS`; очистка выполняется фоново, статус и результат истекают вместе.
  Ожидающие и выполняющиеся задачи не истекают. Большие ColBERT-ответы требуют RAM.
- Перезапуск контейнера удаляет очередь, все task_id, статусы и результаты.
  Сохранения на диск и повторной доставки нет. Старые ID после запуска дают `404`.
- Если воркер аварийно завершился, незавершённые задачи помечаются `failed`
  с кодом `worker_exited`, readiness и новые запросы возвращают `503`.
  Автоматического перезапуска одного воркера нет: выполните `just restart api`.
  Docker healthcheck сам по себе не перезапускает unhealthy-контейнер.
- При штатной остановке API перестаёт принимать задачи, продолжает получать результаты
  и ждёт воркер до `WORKER_SHUTDOWN_TIMEOUT_SECONDS`, затем завершает его принудительно.
  Незавершённые задачи не восстанавливаются.

## Настройки

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `INFERENCE_DEVICE` | `cpu` | `cpu`, `amd`, `amd-wsl` |
| `INFERENCE_BATCH_SIZE` | `8` | Внутренний батч модели, независимо от размера HTTP-батча |
| `USE_FP16` | `false` | FP16 на GPU |
| `QUEUE_CAPACITY` | `128` | Вместимость очереди ожидающих запросов |
| `RESULT_TTL_SECONDS` | `86400` | Хранение завершённых задач и результатов |
| `POLL_INTERVAL_SECONDS` | `0.01` | Интервал ожидания sync-ответа в API |
| `WORKER_STARTUP_TIMEOUT_SECONDS` | `600` | Лимит загрузки модели |
| `WORKER_SHUTDOWN_TIMEOUT_SECONDS` | `30` | Ожидание воркера при остановке |
| `MAX_TEXTS` | `1024` | Максимум строк в запросе |
| `MAX_INPUT_TOKENS` | `8192` | Максимум токенов на строку с учётом `<s>` и `</s>` |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Публикация порта Docker |

`MODEL_ID=BAAI/bge-m3`, ревизия модели закреплена в `.env.example`.
Настройки читаются из `.env` и переменных окружения. Новый `.env.example` не
перезаписывает пользовательский `.env`. При миграции старые REDIS/Celery-настройки
больше не используются; замените `COMPOSE_PROFILES` на `COMPOSE_FILE` по примерам выше.

## REST API

Базовый URL: `http://localhost:8000/api/v1`.
Одинаковый запрос для sync и async:

```json
{
  "model": "baai/bge-m3",
  "input": ["Первый текст", "Второй текст"],
  "embedding_types": ["dense", "sparse", "colbert"]
}
```

`model` и `input` обязательны. Принимается только `baai/bge-m3` без учёта регистра;
значение нормализуется перед отправкой воркеру. `input` — от 0 до 1024 строк.
Пустые батчи и строки допустимы. Если хотя бы одна строка превышает лимит токенов,
весь запрос отклоняется с `422` до очереди, без усечения входа.
`embedding_types` — непустая комбинация без повторов; по умолчанию `["dense"]`.

Необязательный `encoding_format` принимает `float` (по умолчанию) и `base64`,
но пока игнорируется и не передаётся процессу инференса. Оба значения возвращают
эмбеддинги как массивы чисел, в том числе для асинхронного результата.
Это временное отклонение от [контракта OpenRouter](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings);
полноценная поддержка форматов записана в [TODO.md](TODO.md).

| Метод | Поведение |
| --- | --- |
| `POST /api/v1/embeddings` | Ждёт результат, `200` либо ошибка; без раннего `202` |
| `POST /api/v1/embeddings/async` | `202` с `task_id`, `status_url`, `result_url` и временными полями |
| `GET /api/v1/tasks/{task_id}` | Статус задачи; неизвестная/истёкшая задача — `404` |
| `GET /api/v1/tasks/{task_id}/result` | `202` пока ожидается, `200` при успехе, `422` при ошибке |
| `GET /health/live` | Процесс API отвечает |
| `GET /health/ready` | Модель загружена и процесс инференса жив |

Ответы `202` содержат `Location` и `Retry-After: 1`.
Синхронный запрос ждёт завершения задачи без серверного timeout; сетевой timeout
клиента настраивается отдельно. Ошибки валидации и инференса возвращают `422`,
недоступность воркера/полная очередь — `503`, неправильный UUID — `422`.

Пример dense-ответа (вектор сокращён):

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "BAAI/bge-m3",
  "embedding_types": ["dense"],
  "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}]
}
```

`data[].embedding` — нормализованный dense-вектор из 1024 чисел.
`sparse` — объект `{ "token_id": weight }`, ключи — строковые ID токенов.
`colbert` — массив векторов токенов в формате FlagEmbedding, без CLS и padding.
Невыбранные типы отсутствуют. `index` сохраняет порядок строк.

**Текущие расширения и отклонения от OpenRouter:**

- `embedding_types`, `task_id`, асинхронный метод и `/api/v1/tasks/*` — расширения Karavan.
- Поддерживается только массив строк в `input`, а не все форматы OpenRouter.
- `data[].embedding` и `data[].object: "embedding"` соответствуют формату OpenRouter.
  `sparse` и `colbert` — расширения. Без типа `dense` поле `embedding` отсутствует.
- Поля верхнего уровня `object` и `usage` пока не реализованы.
- Валидация возвращает FastAPI `422` с `detail`, тогда как OpenRouter документирует
  `400` с `error.code` и `error.message` для неверных параметров
  ([ошибки OpenRouter](https://openrouter.ai/docs/api/reference/errors-and-debugging)).
- Регистронезависимость `model` — поведение Karavan; для POST OpenRouter она не подтверждена.

### Пример в PowerShell

```powershell
$body = @{ model = 'baai/bge-m3'; input = @('Привет, мир') } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/embeddings `
    -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

## Разработка и проверки

```powershell
just setup
just test
just lint
just config
just smoke
```

`just setup` устанавливает API и dev-зависимости без Torch и загрузки модели.
Для локального CPU-инференса: `uv sync --locked --extra api --extra cpu`,
затем `uv run --no-sync python -m api`.
CPU и AMD extras взаимоисключающие. После правки зависимостей — `just lock`.
Все Docker-установки используют lock-файлы; AMD WSL использует отдельный lock,
который обновляется командой:

```text
uv pip compile --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --excludes requirements/amd-wsl-provided.txt requirements/amd-wsl.in -o requirements/amd-wsl.lock
```

Unit-тесты не скачивают модель. Есть тесты настоящего `spawn` и IPC с тестовым
движком: FIFO, большие ответы, ошибка задачи, падение процесса, startup failure,
переполнение очереди и ограниченная по времени остановка.
`just smoke` проверяет реальную модель через API.
`just recovery` перезапускает контейнер, проверяет потерю старых ID и успешность
нового инференса; запускайте отдельно от клиентской нагрузки.

- `duper_test/` — измерение HTTP-батчей dense.
- `super_test/api_capacity.py` — параллельные HTTP-запросы, все комбинации типов.
- `puper_test/` — прямой тест внутренних батчей модели. Перед ним остановите `api`,
  чтобы освободить GPU; тест запускается отдельной явной командой.

## Структура

```text
api/             HTTP API, жизненный цикл GPU-процесса и сбор результатов
gpu_worker/      загрузка модели и цикл обработки IPC-задач
common/          общие схемы, настройки, токенизатор, хранилище в памяти
Dockerfile       единый Dockerfile: cpu / amd / amd-wsl
compose.yaml     один контейнер api
compose.amd*.yaml доступ к GPU для соответствующей платформы
requirements/    lock-файл зависимостей AMD WSL
scripts/         smoke, restart и проверка GPU
```

Старые контейнеры предыдущей архитектуры удаляются командой
`docker compose up -d --build --remove-orphans`. Старый volume Redis не используется;
данные в нём автоматически не удаляются.
