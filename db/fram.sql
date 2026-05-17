CREATE EXTENSION q3c;

-- Image storage metadata
DROP TABLE IF EXISTS images CASCADE;
CREATE TABLE images (
       id SERIAL PRIMARY KEY,
       filename TEXT UNIQUE,
       night TEXT,
       time TIMESTAMP,
       target INT,
       type TEXT,
       filter TEXT,
       exposure FLOAT,
       ccd TEXT,
       serial INT,
       binning TEXT,
       site TEXT,
       ra FLOAT,
       dec FLOAT,
       radius FLOAT,
       width INT,
       height INT,
       cropped_width INT,
       cropped_height INT,
       footprint POLYGON,
       footprint10 POLYGON,
       mean FLOAT,
       median FLOAT,
       keywords JSONB
);

CREATE INDEX ON images(filename);
CREATE INDEX ON images(night);
CREATE INDEX ON images(time);
CREATE INDEX ON images(target);
CREATE INDEX ON images(type);
CREATE INDEX ON images(filter);
CREATE INDEX ON images(ccd);
CREATE INDEX ON images(serial);
CREATE INDEX ON images(site);
CREATE INDEX ON images(binning);

CREATE INDEX images_q3c_idx ON images (q3c_ang2ipix(ra, dec));

-- For faster ilike searches on OBJECT header field
CREATE INDEX ON images(upper(keywords->>'OBJECT') text_pattern_ops);

-- Dedicated view for calibration frames only
CREATE OR REPLACE VIEW calibrations AS
SELECT *
FROM images
WHERE type='masterdark' OR type='bias' OR type='dcurrent' OR type='masterflat';


-- Stats as materialized views
-- timeline
DROP MATERIALIZED VIEW IF EXISTS image_stats_daily_site;
CREATE MATERIALIZED VIEW image_stats_daily_site AS
SELECT
      night,
      substr(night, 1, 6) AS month,
      site,
      count(*)::bigint AS nimages,
      min(time) AS first_time,
      max(time) AS last_time
FROM images
WHERE night IS NOT NULL
GROUP BY night, substr(night, 1, 6), site
ORDER BY night, site;

-- site stats
DROP MATERIALIZED VIEW IF EXISTS image_stats_site;
CREATE MATERIALIZED VIEW image_stats_site AS
SELECT
      site,
      count(*)::bigint AS nimages,
      min(night) AS first_night,
      max(night) AS last_night,
      min(time) AS first_time,
      max(time) AS last_time
FROM images
GROUP BY site
ORDER BY site;

-- site-ccd-type-filter
DROP MATERIALIZED VIEW IF EXISTS image_stats_type;
CREATE MATERIALIZED VIEW image_stats_type AS
SELECT
        site, ccd, serial, type, filter,
        count(1) as nimages
FROM images
GROUP BY site, ccd, serial, type, filter
ORDER BY site, ccd, serial, type, filter;
