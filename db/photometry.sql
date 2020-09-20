-- Photometry results
DROP TABLE IF EXISTS photometry;
CREATE TABLE photometry (
--       image INT REFERENCES images (id) ON DELETE CASCADE,
       image INT,
       time TIMESTAMP,

       night TEXT,
       site TEXT,
       ccd TEXT,
       filter TEXT,

       ra FLOAT,
       dec FLOAT,
       mag FLOAT,
       magerr FLOAT,
       flags INT,
       std FLOAT,
       nstars INT,
       fwhm FLOAT
);

CREATE INDEX ON photometry (q3c_ang2ipix(ra, dec));
CREATE INDEX ON photometry (image);
