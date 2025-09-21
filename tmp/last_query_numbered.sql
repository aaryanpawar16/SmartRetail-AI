   1: -- src/queries/extract_insights.sql
   2: -- Uses AI.GENERATE_TABLE to parse raw call text into structured insights
   3: -- Replace coral-hull-470715-m0.your_dataset at runtime with your project.dataset
   4: 
   5: SELECT *
   6: FROM AI.GENERATE_TABLE(
   7:   MODEL 'default',
   8:   -- Input prompt per row. `call_text` is the raw transcript.
   9:   -- The model should output rows with columns matching the schema below.
  10:   (
  11:     SELECT call_id, call_text
  12:     FROM `coral-hull-470715-m0.your_dataset.support_calls`
  13:     ORDER BY call_timestamp
  14:     LIMIT 1000
  15:   ),
  16:   -- Schema: the fields we want returned.
  17:   STRUCT<
  18:     call_id INT64,
  19:     summary STRING,
  20:     sentiment STRING,           -- e.g., positive/neutral/negative
  21:     priority STRING,            -- e.g., low/medium/high
  22:     tags ARRAY<STRING>,         -- short labels
  23:     action_items ARRAY<STRING>  -- actionable next steps
  24:   >()
  25: )
