-- Example SQL that uses ML.GENERATE_TEXT to produce short marketing emails
SELECT
customer_id,
email,
ML.GENERATE_TEXT(MODEL `region-us`.default_text_model,
CONCAT('Write a friendly, 60-word marketing email recommending ', product_name,
' to a customer whose last purchase category was ', last_category, '. Include a clear CTA.')) AS email_body
FROM
`PROJECT.DATASET.customer_history`
LIMIT 200;