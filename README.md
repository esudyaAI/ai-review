# jobbytime-ai-review

AI-ревьюер merge request'ов для репозиториев группы Jobbytime.

## Как подключить

В `.gitlab-ci.yml` целевого репозитория:

```yaml
include:
  - project: jobbytime.com-group/jobbytime-ai-review
    ref: v1
    file: templates/ai-review.yml
    inputs:
      allow_failure: true   # shadow-режим; снять после обкатки
```

Нужные переменные CI (уровень группы, Masked, без Protected):

- `OPENAI_API_KEY`
- `AI_REVIEW_GITLAB_TOKEN`

Настройки конкретного репозитория — в файле `.ai-review.yml` в его корне.

## Коды выхода

| Код | Значение |
| --- | --- |
| 0 | замечаний нет, merge не блокируется |
| 1 | есть открытые замечания |
| 2 | техническая ошибка, нужен реран |
| 3 | merge request слишком большой, надо разбить |
