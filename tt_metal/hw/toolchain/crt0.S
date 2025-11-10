
	.section .start,"ax",@progbits
	.global _start
	.type _start,@function
	/* s1 is coordinates.  */
_start:

	/* Compute TP and SP at either end of the locals */
	lui	tp,%hi(__local_base)
	addi	tp,tp,%lo(__local_base)
	lui	sp,%hi(__local_stride)
	addi	sp,sp,%lo(__local_stride)
	csrr	s2,mhartid
	mul	s2,sp,s2
	add	tp,tp,s2
	add	sp,tp,sp

	/* FIXME: have a better API than regular main.  */
	addi	sp,sp,-16
	addi	a0,sp,8
	sw	a0,0(sp) // argv[0]
	sw	zero,4(sp) // argv[1]
	sw	s1,8(sp) // argv[0][0..3]
	sw	zero,12(sp) // argv[0][4..7]

	/* FIXME: Need to copy .data, zero .bss exactly once (or have the loading process do this?)  */
	/* FIXME: Need to copy .tdata, zero .tbss on every thread.  */
	/* FIXME: Where are the initial .data and .tdata images placed?  */
	call main
	tail exit

	.size _start, .-_start
