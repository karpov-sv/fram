CREATE EXTENSION q3c;

-- Image storage metadata
DROP TABLE IF EXISTS images CASCADE;
CREATE TABLE images (
       id SERIAL PRIMARY KEY,
       filename TEXT,
       night TEXT,
       time TIMESTAMP,
       target INT,
       type TEXT,
       filter TEXT,
       ccd TEXT,
       ra FLOAT,
       dec FLOAT,
       radius FLOAT,
       exposure FLOAT,
       width INT,
       height INT,
       mean FLOAT,
       median FLOAT,
       keywords JSONB
);
CREATE INDEX ON images(filename);
CREATE INDEX ON images(night);
CREATE INDEX ON images(time);
CREATE INDEX ON images(type);
CREATE INDEX ON images(filter);
CREATE INDEX ON images(ccd);
CREATE INDEX images_q3c_idx ON images (q3c_ang2ipix(ra, dec));
