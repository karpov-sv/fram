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

-- Dedicated view for calibration frames only
CREATE OR REPLACE VIEW calibrations AS
SELECT *
FROM images
WHERE type='masterdark' OR type='bias' OR type='dcurrent' OR type='masterflat';
