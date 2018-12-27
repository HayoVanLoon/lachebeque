CREATE TABLE
  tmp.jokes_test AS
SELECT
  id,
  text,
  score,
  norm_score,
  norm_log_score
FROM
  jokes.reddit2
WHERE
  score > 0
  AND LENGTH(text) < 400
  AND MOD(ABS(FARM_FINGERPRINT(text)), 4) = 0