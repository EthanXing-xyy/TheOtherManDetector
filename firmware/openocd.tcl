source [find interface/kitprog3.cfg]

transport select swd


source [find target/infineon/cy8cxxa.cfg]
${TARGET}.cm4 configure -rtos auto -rtos-wipe-on-reset-halt 1
psoc6 sflash_restrictions 1
