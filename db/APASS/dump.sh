#!/bin/sh

echo "DROP TABLE IF EXISTS apass;" | psql fram

echo "CREATE TABLE apass (ra FLOAT, dec FLOAT, B FLOAT, V FLOAT, G FLOAT, R FLOAT, I FLOAT, var INT DEFAULT 0, Berr FLOAT, Verr FLOAT, Gerr FLOAT, Rerr FLOAT, Ierr FLOAT);" | psql fram

unzip -c apass_dr9a.zip|awk '{if($1*1>0) print $2","$4","$10","$8","$11","$12","$13","$16","$14","$17","$18","$19}'|psql fram -c "COPY apass (ra,dec,B,V,G,R,I,Berr,Verr,Gerr,Rerr,Ierr) FROM stdin CSV;"
unzip -c apass_dr9b.zip|awk '{if($1*1>0) print $2","$4","$10","$8","$11","$12","$13","$16","$14","$17","$18","$19}'|psql fram -c "COPY apass (ra,dec,B,V,G,R,I,Berr,Verr,Gerr,Rerr,Ierr) FROM stdin CSV;"
# unzip -c apass_dr9a.zip|awk '{if($1*1>0 && $4 > -30 && $8 > 0 && $8<15) print $2","$4","$10","$8","$11","$12","$13","$16","$14","$17","$18","$19}'|psql fram -c "COPY apass (ra,dec,B,V,G,R,I,Berr,Verr,Gerr,Rerr,Ierr) FROM stdin CSV;"
# unzip -c apass_dr9b.zip|awk '{if($1*1>0 && $4 > -30 && $8 > 0 && $8<15) print $2","$4","$10","$8","$11","$12","$13","$16","$14","$17","$18","$19}'|psql fram -c "COPY apass (ra,dec,B,V,G,R,I,Berr,Verr,Gerr,Rerr,Ierr) FROM stdin CSV;"

echo "CREATE INDEX ON apass (q3c_ang2ipix(ra,dec));" | psql fram
# echo "CREATE INDEX ON apass (B);" | psql fram
# echo "CREATE INDEX ON apass (V);" | psql fram
# echo "CREATE INDEX ON apass (R);" | psql fram
# echo "CREATE INDEX ON apass (G);" | psql fram
# echo "CREATE INDEX ON apass (I);" | psql fram

echo "UPDATE apass p SET var = 1 FROM vsx v WHERE q3c_join(v.ra, v.dec, p.ra, p.dec, 0.01);" | psql fram
