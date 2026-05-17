-- Photometry results
DROP TABLE IF EXISTS photometry_all;
CREATE TABLE photometry_all (
--       image INT REFERENCES images (id) ON DELETE CASCADE,
       image INT,
       time TIMESTAMP NOT NULL,
       filter CHAR NOT NULL,

       ra FLOAT,
       dec FLOAT,

       mag FLOAT, -- calibrated magnitude fitted without color term
       magerr FLOAT, -- magnitude error including zero point error
       flags INT,

       mag_color FLOAT, -- calibrated magnitude fitted with color term
       color_term FLOAT, -- color term

       std FLOAT,
       zp_std FLOAT,
       nstars INT,
       final_frac FLOAT,

       fwhm FLOAT
);

-- The table is supposed to be inherited by manually selected shards
-- corresponding to individual sites, cameras or configurations

-- Indices are to be created later, after bulk populating the tables
-- CREATE INDEX ON photometry_all (q3c_ang2ipix(ra, dec));
-- CREATE INDEX ON photometry_all (image);

-- Examples of use:
-- CREATE TABLE photometry_auger_nf4 (LIKE photometry_all); -- create sharded sub-table
-- ALTER TABLE photometry_auger_nf4 INHERIT LIKE photometry_all; -- connect populated sub-table to main one
