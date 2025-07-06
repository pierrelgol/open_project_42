.section .entry, "ax"
.global _start

_start:
    movi sp, _stack_top
    
    movi a2, _data_start
    movi a3, _data_end
    movi a4, _data_load_start
    
    beq a2, a3, data_done

data_copy:
    l32i a5, a4, 0
    s32i a5, a2, 0
    addi a2, a2, 4
    addi a4, a4, 4
    blt a2, a3, data_copy

data_done:
    
    movi a2, _bss_start
    movi a3, _bss_end
    movi a4, 0
    
bss_init:
    beq a2, a3, bss_done
    s32i a4, a2, 0
    addi a2, a2, 4
    j bss_init

bss_done:
    
    movi a2, _bss_end          
    movi a3, _stack_bottom     
    movi a4, 0                 
    
heap_init:
    beq a2, a3, heap_done
    s32i a4, a2, 0            
    addi a2, a2, 4
    j heap_init

heap_done:    
    call0 kernel_main

halt:
    j halt
