/* ESP32 Kernel Linker Script - Updated for more accurate ESP32 memory layout */

/* esp32_technical_reference_manual_en.pdf 1.3.2 */
MEMORY {
  drom   (rw)   : ORIGIN = 0x3ff90000, LENGTH = 64K
  irom   (rwx)  : ORIGIN = 0x40000000, LENGTH = 384K

  cache0 (rwx)  : ORIGIN = 0x40070000, LENGTH = 32K
  cache1 (rwx)  : ORIGIN = 0x40078000, LENGTH = 32K
  iram   (rwx)  : ORIGIN = 0x40080400, LENGTH = 127K  /* First 1k is vectors */
  dram   (rw)   : ORIGIN = 0x3ffb0000, LENGTH = 320K

  dflash (rw)   : ORIGIN = 0x3f400000, LENGTH = 1024K
  psram  (rw)   : ORIGIN = 0x3f800000, LENGTH = 1024K
  iflash (rwx)  : ORIGIN = 0x400c2000, LENGTH = 11512K
}

_eram = ORIGIN(dram) + LENGTH(dram);
ENTRY(_start)

SECTIONS {
  /* Entry point and text section */
  .entry : ALIGN(4) {
    KEEP(*(.entry))
  } > iram

  .text : ALIGN(4) {
    *(.text*)
    *(.text)
  } > iram

  .data : ALIGN(4) {
    _data_start = .;
    *(.data)
    *(.data*)
    *(.rodata)
    *(.rodata*)
    *(.gnu.linkonce.r.*)
    *(.rodata1)
    . = ALIGN(4);
    _data_end = .;
  } > dram AT> irom

  /* Data load address in flash */
  _data_load_start = LOADADDR(.data);

  .bss : ALIGN(4) {
    _bss_start = .;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _bss_end = .;
  } > dram

  /* Stack section */
  .stack : ALIGN(4) {
    _stack_bottom = .;
    . = . + 0x10000;  /* 64KB stack */
    _stack_top = .;
  } > dram

  . = ALIGN(4);
  PROVIDE(end = .);
  PROVIDE(_end = .);

  /*
  /DISCARD/ : { *(.debug) *(.debug*) *(.xtensa.*) *(.comment) }
  */
}
