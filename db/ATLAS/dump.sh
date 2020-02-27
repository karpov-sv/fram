#!/bin/bash

echo "DROP TABLE IF EXISTS atlas;" | psql fram

echo "CREATE TABLE atlas (ra FLOAT, dec FLOAT, Gaia FLOAT, dGaia FLOAT, BP FLOAT, dBP FLOAT, RP FLOAT, dRP FLOAT, g FLOAT, dg FLOAT, r FLOAT, dr FLOAT, i FLOAT, di FLOAT, z FLOAT, dz FLOAT);" | psql fram

tar xjvf hlsp_atlas-refcat2_atlas_ccd_00-m-16_multi_v1_cat.tbz -O|awk 'BEGIN{OFMT="%.8f";FS=",";OFS=","}{print $1/1e8, $2/1e8, $9/1e3, $10/1e3, $11/1e3, $12/1e3, $13/1e3, $14/1e3, $22/1e3, $23/1e3, $26/1e3, $27/1e3, $30/1e3, $31/1e3, $34/1e3, $35/1e3}' | psql fram -c "COPY atlas FROM stdin WITH DELIMITER AS ','"

echo "CREATE INDEX atlas_q3c_idx ON atlas (q3c_ang2ipix(ra,dec));" | psql fram
