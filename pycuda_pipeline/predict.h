/* Header to test of C modules for arrays for Python: C_test.c */
#include "complex.h"

double *D_ptr(PyArrayObject *arrayin)  {
	return (double *) arrayin->data;
}

int *I_ptr(PyArrayObject *arrayin)  {
	return (int *) arrayin->data;
}


double complex *DC_ptr(PyArrayObject *arrayin)  {
	return (double complex *) arrayin->data;
}


void *Prod22(double complex *a, double complex *b, double complex *c)
{
  double complex a00, a01, a10, a11;
  double complex b00, b01, b10, b11;

  a00=a[0]; a10=a[2]; a01=a[1]; a11=a[3];
  b00=b[0]; b10=b[2]; b01=b[1]; b11=b[3];

  c[0]=a00*b00+a01*b10;
  c[1]=a00*b01+a01*b11;
  c[2]=a10*b00+a11*b10;
  c[3]=a10*b01+a11*b11;
  return c;

}

void *Prod22H(double complex *a, double complex *b, double complex *c)
{
  double complex a00, a01, a10, a11;
  double complex b00, b01, b10, b11;

  a00=a[0]; a10=a[2]; a01=a[1]; a11=a[3];
  b00=conj(b[0]); b10=conj(b[1]); b01=conj(b[2]); b11=conj(b[3]);

  c[0]=a00*b00+a01*b10;
  c[1]=a00*b01+a01*b11;
  c[2]=a10*b00+a11*b10;
  c[3]=a10*b01+a11*b11;
  return c;

}





static PyObject *predict(PyObject *self, PyObject *args);
static PyObject *DotParallel(PyObject *self, PyObject *args);
static PyObject *BatchProd(PyObject *self, PyObject *args);
static PyObject *predictSols(PyObject *self, PyObject *args);
static PyObject *predictSolsPol(PyObject *self, PyObject *args);
static PyObject *predictSolsPolCluster(PyObject *self, PyObject *args);
static PyObject *predictParallel(PyObject *self, PyObject *args);
static PyObject *predictParallelReverse(PyObject *self, PyObject *args);

