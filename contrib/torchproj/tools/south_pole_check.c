/* Reproduces the asymmetric latitude over-range check in src/fwd.cpp:61.
 *
 * Build against this tree (no proj.db needed):
 *   cmake <PROJ> -DEXE_SQLITE3=/bin/true -DENABLE_TIFF=OFF -DENABLE_CURL=OFF \
 *         -DBUILD_TESTING=OFF -DBUILD_APPS=OFF && make proj
 *   cc south_pole_check.c -I<PROJ>/src -I<build>/src -L<build>/lib -lproj -lm
 *
 * Expected with this tree: -pi/2 - 1e-15 fails (inf), +pi/2 + 1e-15 clamps.
 * PROJ 9.8.1 clamps both.
 */
#include <math.h>
#include <proj.h>
#include <stdio.h>
int main(void) {
    PJ *P = proj_create(PJ_DEFAULT_CTX, "+proj=cart +ellps=WGS84");
    if (!P) {
        printf("create failed\n");
        return 1;
    }
    double phis[] = {-M_PI_2,         -M_PI_2 - 1e-15, -M_PI_2 - 5e-13,
                     -M_PI_2 - 2e-12, M_PI_2 + 1e-15,  M_PI_2 + 5e-13,
                     M_PI_2 + 2e-12};
    for (int i = 0; i < 7; i++) {
        PJ_COORD c = proj_coord(0.3, phis[i], 0, 0);
        PJ_COORD r = proj_trans(P, PJ_FWD, c);
        printf("phi=%.17g -> x=%g y=%g z=%.10g\n", phis[i], r.xyz.x, r.xyz.y,
               r.xyz.z);
    }
    proj_destroy(P);
    return 0;
}
