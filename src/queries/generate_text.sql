-- Example: use ML.GENERATE_TEXT to produce a product recommendation description
-- Replace `project.dataset.table` with your table


SELECT
product_id,
ML.GENERATE_TEXT(
MODEL `region-us`.default_text_model,
CONCAT('Write a short, persuasive product description for: ', product_name, '. Use an upbeat tone and include one bullet point feature.')
) AS product_description
FROM
`PROJECT.DATASET.products`
LIMIT 50;