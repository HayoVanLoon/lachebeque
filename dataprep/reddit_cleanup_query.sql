CREATE TABLE
  jokes.reddit2 AS

WITH
  max_score AS (SELECT MAX(score) FROM `incentro-tpu-parlor.jokes.reddit`)
SELECT
  id,
  REGEXP_REPLACE(REGEXP_REPLACE(LOWER(CONCAT(title, ' ', body)), r'\n|(\.\.\.)', ' '), r'\s+', ' ') text,
  score,
  score / (SELECT * FROM max_score) norm_score,
  IF(score > 0, LOG(score) / LOG((SELECT * FROM max_score)), NULL) norm_log_score,
  title,
  body
FROM
  `incentro-tpu-parlor.jokes.reddit`

