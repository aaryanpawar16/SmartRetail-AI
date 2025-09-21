-- src/queries/extract_insights.sql
-- Uses AI.GENERATE_TABLE to parse raw call text into structured insights
-- Replace PROJECT.DATASET at runtime with your project.dataset

SELECT *
FROM AI.GENERATE_TABLE(
  MODEL 'default',
  -- Input prompt per row. `call_text` is the raw transcript.
  -- The model should output rows with columns matching the schema below.
  (
    SELECT call_id, call_text
    FROM `PROJECT.DATASET.support_calls`
    ORDER BY call_timestamp
    LIMIT 1000
  ),
  -- Schema: the fields we want returned.
  STRUCT<
    call_id INT64,
    summary STRING,
    sentiment STRING,           -- e.g., positive/neutral/negative
    priority STRING,            -- e.g., low/medium/high
    tags ARRAY<STRING>,         -- short labels
    action_items ARRAY<STRING>  -- actionable next steps
  >()
)
