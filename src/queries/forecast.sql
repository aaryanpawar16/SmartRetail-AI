SELECT *
FROM AI.FORECAST(
  (
    SELECT DATE(order_date) AS time_day,
           SUM(`sales_amount`) AS sales
    FROM `PROJECT.DATASET.sales`
    GROUP BY time_day
  ),
  timestamp_col => 'time_day',
  data_col => 'sales',
  horizon => 7
);
