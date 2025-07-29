# set SFPI release version information

sfpi_version=v6.13.0-renumber
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=f5ce7af5a8e7e11da55447123911e094
sfpi_x86_64_Linux_deb_md5=33f48fbfb463783def9b8885ce80810a
sfpi_x86_64_Linux_rpm_md5=b6cf8e41793b18cb275a72493e72ffb2
