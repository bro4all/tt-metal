# set SFPI release version information

sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi_\([-.0-9a-zA-Z]*\)_\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_version=\2'"\n"'sfpi_\3_\4_md5=\1/' *-build/sfpi_*.md5 | sort -u
sfpi_version=7.3.0-ttinsn2
sfpi_x86_64_deb_md5=f49ddc25025f24326270ac8750351a79
sfpi_x86_64_rpm_md5=d9506bd34ca95a293da88165db2e643a
sfpi_x86_64_txz_md5=f2de178ec9b0d8b7f5b83357a9453d0d
