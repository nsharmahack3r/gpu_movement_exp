# Taste

## Workflow
- Prefers small, dedicated, self-contained scripts over inline `python -c` one-liners (e.g. in the README), so the behavior is easy to edit. Confidence: 0.8
- Uses ntfy.sh push notifications to monitor long-running GPU experiments, wanting a notification after each experiment completes (plus on start/failure/finish). Already keeps an `NTFY_NOTIFICATION_TOPIC` env var in `.env`/`.sample.env`. Confidence: 0.6
- Cares about the end-user presentation of notification output — wants phone notifications to render as proper formatted notifications (title, tags, etc.), not raw JSON/payloads. Confidence: 0.5
