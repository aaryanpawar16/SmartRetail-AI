-- Example: Create a structured table of insights from raw support logs using AI.GENERATE_TABLE
-- AI.GENERATE_TABLE will return a table according to the schema you define in the prompt


WITH logs AS (
SELECT
support_id,
timestamp,
customer_id,
transcript
FROM
`PROJECT.DATASET.support_calls`
LIMIT 200
)


SELECT * FROM
AI.GENERATE_TABLE(
MODEL "default",
(
SELECT AS STRUCT
STRING_AGG(CONCAT('transcript: ', transcript, '\n'), '\n') AS combined
FROM logs
),
STRUCT(
'analysis' AS prompt,
ARRAY<STRUCT<name STRING, type STRING>>[
('support_id','STRING'),
('category','STRING'),
('sentiment','STRING'),
('action_items','STRING')
] AS schema
)
);