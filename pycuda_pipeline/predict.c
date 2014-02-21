/* A file to test imorting C modules for handling arrays to Python */

#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include "complex.h"
#include "predict.h"
#include <assert.h>
#include <stdio.h>

/* #### Globals #################################### */

/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */

/* ==== Set up the methods table ====================== */
static PyMethodDef predict_Methods[] = {
	{"predict", predict, METH_VARARGS},
	{"predictSols", predictSols, METH_VARARGS},
	{"predictSolsPol", predictSolsPol, METH_VARARGS},
	{"predictSolsPolCluster", predictSolsPolCluster, METH_VARARGS},
	{"DotParallel", DotParallel, METH_VARARGS},
	{"BatchProd", BatchProd, METH_VARARGS},
	{"predictParallel", predictParallel, METH_VARARGS},
	{"predictParallelReverse", predictParallelReverse, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void initpredict()  {
	(void) Py_InitModule("predict", predict_Methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}


static PyObject *predictSolsPolCluster(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyArrayObject *NpVisIn, *NpUVWin, *NpLMin, *matout,*NpLM,
    *NpWaveL,*NpSols,*NpTimesSols,*NpA0,*NpA1,*NpInfo,*NpCluster;
  int *A0,*A1;//, *Cluster;
  int* Cluster;
  double *UVWin,*LM,*WaveL,*TimesSols,*Times, *Info;
  double complex  *VisIn, *Sols;
  int nrow,npol,nsources,i,dim[2];

  
  if (!PyArg_ParseTuple(args, "OO!O!O!O!O!O!O!O!", 
			&ObjVisIn, 
			&PyArray_Type, &NpA0, 
			&PyArray_Type, &NpA1, 
			&PyArray_Type, &NpUVWin, 
			&PyArray_Type, &NpLM,
			&PyArray_Type, &NpWaveL, 
			&PyArray_Type, &NpSols,
			&PyArray_Type, &NpCluster,
			&PyArray_Type, &NpInfo))  return NULL;
  
  NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX128, 0, 3);
  //NpUVWin = (PyArrayObject *) PyArray_ContiguousFromObject(ObjUVWin, PyArray_DOUBLE, 0, 3);
  //NpLM = (PyArrayObject *) PyArray_ContiguousFromObject(ObjLM, PyArray_DOUBLE, 0, 3);
  
  //  printf("*UVWin +1j\n");
  VisIn = DC_ptr(NpVisIn);
  Sols  = DC_ptr(NpSols);
  Info  = D_ptr(NpInfo);
  double RefWave=Info[0];

  A0=I_ptr(NpA0);
  A1=I_ptr(NpA1);
  Cluster=I_ptr(NpCluster);

  printf("%i %i %i\n",A0[0],A1[0],Cluster[0]);

  UVWin = D_ptr(NpUVWin);
  LM    = D_ptr(NpLM);
  WaveL = D_ptr(NpWaveL);
  
  int ch,dd,ddCluster,nchan,ndir,na;
  nrow=NpVisIn->dimensions[0];
  nchan=NpVisIn->dimensions[1];
  na=NpSols->dimensions[2];


  int nlm;
  ndir=NpLM->dimensions[0];
  nlm=NpLM->dimensions[1];

  /* Do the calculation. */
  double phase,l,m,n,u,v,w;
  double complex c0,c1,result;
  c0=2.*3.141592*I;
  double complex *p0;
  double *p1;
  p0=VisIn;
  p1=UVWin;
  
  double complex *J0, *J1;
  double complex JJ[4],Sky[4];


  double ThisFlux, fI, fQ, fU, fV, Alpha;

  for(dd=0;dd<ndir;dd++){
    l=LM[0];
    m=LM[1];
    fI=LM[2];
    Alpha=LM[3];
    fQ=LM[4];
    fU=LM[5];
    fV=LM[6];

    ddCluster=Cluster[dd];
    //printf("3 dir=%i ClusterId=%i\n",dd,ddCluster);
    Sky[0]=(fI+fQ);
    Sky[1]=(fU+I*fV);
    Sky[2]=(fU-I*fV);
    Sky[3]=(fI-fQ);

    LM+=nlm;
    n=sqrt(1.-l*l-m*m)-1.;
    //printf("====================================\n");
    //printf("dd: %i/%i nchan=%i nrow=%i (l,m,s,al)=(%f,%f,%f,%f)\n",dd,ndir,nchan,nrow,l,m,fI,Alpha);
    //printf("l,m: %f %f %f\n",l,m,n);
    //printf("*UVWin %f %f %f\n",l,m,n);
    VisIn=p0;
    UVWin=p1;
    for ( i=0; i<nrow; i++)  {
	phase=(*UVWin++)*l;
	//printf("cc %f \n",phase);
	phase+=(*UVWin++)*m;
	//printf("cc %f \n",phase);
	phase+=(*UVWin++)*n;
	//printf("cc %f \n",phase);
	for(ch=0;ch<nchan;ch++){ 
	  //printf("ch: %i %f\n",ch,WaveL[ch]);
	  c1=c0/WaveL[ch];
	  //printf("0 %f\n",c1);
 	  ThisFlux=pow(RefWave/WaveL[ch],Alpha);
	  //printf("1 %f\n",ThisFlux);
	  //result=ThisFlux*cexp(phase*c1);

	  result=ThisFlux*cexp(phase*c1);
	  //printf("2 (%f, 1j %f)\n",creal(result), cimag(result));
	  
	  //Sols shape: Nd, Nf, Na, 4
	  //ddCluster=Cluster[dd];
	  //printf("3 dir=%i ClusterId=%i\n",dd,ddCluster);
	  J0=Sols+ ddCluster*na*nchan*4 + ch*na*4 + A0[i]*4;
	  J1=Sols+ ddCluster*na*nchan*4 + ch*na*4 + A1[i]*4;
	  //printf("4\n");

	  

	  //	  Prod22H(J0,J1,JJ);
	  Prod22(J0,Sky,JJ);
	  Prod22H(JJ,J1,JJ);


	  // printf
	  /* int pol; */
	  /* for(pol=0; pol<4; pol++){ */
	  /*   printf("J0[%i] = %f + i%f\n", pol, creal(J0[pol]), cimag(J0[pol])); */
	  /* } */
	  /* printf("\n"); */
	  /* for(pol=0; pol<4; pol++){ */
	  /*   printf("J1[%i] = %f + i%f\n", pol, creal(J1[pol]), cimag(J1[pol])); */
	  /* } */
	  /* printf("\n"); */
	  /* for(pol=0; pol<4; pol++){ */
	  /*   printf("JJ[%i] = %f + i%f\n", pol, creal(JJ[pol]), cimag(JJ[pol])); */
	  /* } */
	  // end printf
  
	  /* printf("=============\n"); */
	  /* printf("dd: %i/%i nchan=%i nrow=%i (l,m,s,al)=(%f,%f,%f,%f)\n",dd,ndir,nchan,nrow,l,m,fI,Alpha); */
 	  /* printf("(phase,ThisFlux,result,abs(result))=( %f, %f, (%f, 1j*%f), %f)\n", */
	  /* 	 phase,ThisFlux,creal(result),cimag(result),cabs(result)); */
 	  /* printf("Jones =[[ (%f+1j*%f), (%f+1j*%f)], [(%f+1j*%f), (%f+1j*%f)]]\n", */
	  /* 	 creal(JJ[0]),cimag(JJ[0]), */
	  /* 	 creal(JJ[1]),cimag(JJ[1]), */
	  /* 	 creal(JJ[2]),cimag(JJ[2]), */
	  /* 	 creal(JJ[3]),cimag(JJ[3])); */
	  //printf("\n");
	  

	  VisIn[0]+=JJ[0]*result;
	  VisIn[1]+=JJ[1]*result;
	  VisIn[2]+=JJ[2]*result;
	  VisIn[3]+=JJ[3]*result;

	  /* VisIn[0]+=result; */
	  /* VisIn[3]+=result; */

	  
 	  /* printf("Jones =[[ (%f+1j*%f), (%f+1j*%f)], [(%f+1j*%f), (%f+1j*%f)]]\n", */
	  /* 	 creal(VisIn[0]),cimag(VisIn[0]), */
	  /* 	 creal(VisIn[1]),cimag(VisIn[1]), */
	  /* 	 creal(VisIn[2]),cimag(VisIn[2]), */
	  /* 	 creal(VisIn[3]),cimag(VisIn[3])); */

	  VisIn+=4;

	  //if(i>10){assert(1==0);}

	}
	
    }
  }
  //assert(1==0);
  //free_Carrayptrs2(UVWin);
  //return Py_BuildValue("OO", NpVisInRe,NpVisInIm);//NpUVWin);
  return PyArray_Return(NpVisIn);
}




static PyObject *predictSolsPol(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyArrayObject *NpVisIn, *NpUVWin, *NpLMin, *matout,*NpLM,
    *NpWaveL,*NpSols,*NpTimesSols,*NpA0,*NpA1,*NpInfo;
  int *A0,*A1;
  double *UVWin,*LM,*WaveL,*TimesSols,*Times, *Info;
  double complex  *VisIn, *Sols;
  int nrow,npol,nsources,i,dim[2];

  
  if (!PyArg_ParseTuple(args, "OO!O!O!O!O!O!O!", 
			&ObjVisIn, 
			&PyArray_Type, &NpA0, 
			&PyArray_Type, &NpA1, 
			&PyArray_Type, &NpUVWin, 
			&PyArray_Type, &NpLM,
			&PyArray_Type, &NpWaveL, 
			&PyArray_Type, &NpSols,
			&PyArray_Type, &NpInfo))  return NULL;
  
  NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX128, 0, 3);
  //NpUVWin = (PyArrayObject *) PyArray_ContiguousFromObject(ObjUVWin, PyArray_DOUBLE, 0, 3);
  //NpLM = (PyArrayObject *) PyArray_ContiguousFromObject(ObjLM, PyArray_DOUBLE, 0, 3);
  
  //  printf("*UVWin +1j\n");
  VisIn = DC_ptr(NpVisIn);
  Sols  = DC_ptr(NpSols);
  Info  = D_ptr(NpInfo);
  double RefWave=Info[0];

  A0=I_ptr(NpA0);
  A1=I_ptr(NpA1);

  UVWin = D_ptr(NpUVWin);
  LM    = D_ptr(NpLM);
  WaveL = D_ptr(NpWaveL);
  
  int ch,dd,nchan,ndir,na;
  nrow=NpVisIn->dimensions[0];
  nchan=NpVisIn->dimensions[1];
  na=NpSols->dimensions[2];


  int nlm;
  ndir=NpLM->dimensions[0];
  nlm=NpLM->dimensions[1];

  /* Do the calculation. */
  double phase,l,m,n,u,v,w;
  double complex c0,c1,result;
  c0=2.*3.141592*I;
  double complex *p0;
  double *p1;
  p0=VisIn;
  p1=UVWin;
  
  double complex *J0, *J1;
  double complex JJ[4],Sky[4];


  double ThisFlux, fI, fQ, fU, fV, Alpha;

  for(dd=0;dd<ndir;dd++){
    l=LM[0];
    m=LM[1];
    fI=LM[2];
    Alpha=LM[3];
    fQ=LM[4];
    fU=LM[5];
    fV=LM[6];

    Sky[0]=(fI+fQ);
    Sky[1]=(fU+I*fV);
    Sky[2]=(fU-I*fV);
    Sky[3]=(fI-fQ);

    LM+=nlm;
    n=sqrt(1.-l*l-m*m)-1.;
    //printf("====================================\n");
    //printf("dd: %i/%i nchan=%i nrow=%i (l,m,s,al)=(%f,%f,%f,%f)\n",dd,ndir,nchan,nrow,l,m,Flux,Alpha);
    //printf("l,m: %f %f %f\n",l,m,n);
    //printf("*UVWin %f %f %f\n",l,m,n);
    VisIn=p0;
    UVWin=p1;
    for ( i=0; i<nrow; i++)  {
	phase=(*UVWin++)*l;
	//printf("cc %f \n",phase);
	phase+=(*UVWin++)*m;
	//printf("cc %f \n",phase);
	phase+=(*UVWin++)*n;
	//printf("cc %f \n",phase);
	for(ch=0;ch<nchan;ch++){ 
	  //printf("ch: %i %f\n",ch,WaveL[ch]);
	  c1=c0/WaveL[ch];
 	  ThisFlux=pow(RefWave/WaveL[ch],Alpha);
	  //result=ThisFlux*cexp(phase*c1);

	  result=ThisFlux*cexp(phase*c1);
	  
	  //Sols shape: Nd, Nf, Na, 4
	  J0=Sols+ dd*na*nchan*4 + ch*na*4 + A0[i]*4;
	  J1=Sols+ dd*na*nchan*4 + ch*na*4 + A1[i]*4;

	  

	  //	  Prod22H(J0,J1,JJ);
	  Prod22(J0,Sky,JJ);
	  Prod22H(JJ,J1,JJ);

	  /* int pol; */
	  /* for(pol=0; pol<4; pol++){ */
	  /*   printf("J0[%i] = %f + i%f\n", pol, creal(J0[pol]), cimag(J0[pol])); */
	  /* } */
	  /* printf("\n"); */
	  /* for(pol=0; pol<4; pol++){ */
	  /*   printf("J1[%i] = %f + i%f\n", pol, creal(J1[pol]), cimag(J1[pol])); */
	  /* } */
	  /* printf("\n"); */
	  /* for(pol=0; pol<4; pol++){ */
	  /*   printf("JJ[%i] = %f + i%f\n", pol, creal(JJ[pol]), cimag(JJ[pol])); */
	  /* } */
  
	  /* printf("=============\n"); */
	  /* printf("dd: %i/%i nchan=%i nrow=%i (l,m,s,al)=(%f,%f,%f,%f)\n",dd,ndir,nchan,nrow,l,m,Flux,Alpha); */
 	  /* printf("cc %f %f %f %fi (abs=%f)\n",phase,ThisFlux,creal(result),cimag(result),cabs(result)); */
	  /* //printf("\n"); */
	  

	  VisIn[0]+=JJ[0]*result;
	  VisIn[1]+=JJ[1]*result;
	  VisIn[2]+=JJ[2]*result;
	  VisIn[3]+=JJ[3]*result;

	  /* VisIn[0]+=result; */
	  /* VisIn[3]+=result; */

	  VisIn+=4;

	  //if(i>10){assert(1==0);}

	}
	
    }
  }
  //assert(1==0);
  //free_Carrayptrs2(UVWin);
  //return Py_BuildValue("OO", NpVisInRe,NpVisInIm);//NpUVWin);
  return PyArray_Return(NpVisIn);
}




static PyObject *predict(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyArrayObject *NpVisIn, *NpUVWin, *NpLMin, *matout,*NpLM, *NpWaveL,*NpFlux;
  double *UVWin,*LM,*WaveL,*Flux;
  double complex  *VisIn;
  int nrow,npol,nsources,i,dim[2];
  
  if (!PyArg_ParseTuple(args, "OO!O!O!O!", &ObjVisIn, &PyArray_Type, &NpUVWin, 
			&PyArray_Type, &NpLM, &PyArray_Type, &NpWaveL, &PyArray_Type, &NpFlux))  return NULL;
  
  NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX128, 0, 3);
  //NpUVWin = (PyArrayObject *) PyArray_ContiguousFromObject(ObjUVWin, PyArray_DOUBLE, 0, 3);
  //NpLM = (PyArrayObject *) PyArray_ContiguousFromObject(ObjLM, PyArray_DOUBLE, 0, 3);
  
  printf("*UVWin +1j\n");
  VisIn=DC_ptr(NpVisIn);
  UVWin=D_ptr(NpUVWin);
  LM=D_ptr(NpLM);
  WaveL=D_ptr(NpWaveL);
  Flux=D_ptr(NpFlux);
  
  int ch,dd,nchan,ndir;
  nrow=NpVisIn->dimensions[0];
  nchan=NpVisIn->dimensions[1];

  ndir=NpLM->dimensions[0];

  /* Get the dimensions of the input */
  
  /* Make a new double matrix of same dims */
  //matout=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
  
  
  /* Do the calculation. */
  double phase,l,m,n,u,v,w;
  double complex c0,c1,result;
  c0=2.*3.141592*I;
  double complex *p0;
  double *p1;
  p0=VisIn;
  p1=UVWin;

    
  for(dd=0;dd<ndir;dd++){
    l=LM[dd];
    m=LM[dd];
    n=sqrt(1.-l*l-m*m)-1.;
    //printf("dd: %i/%i nchan=%i nrow=%i (l,m)=(%f,%f)\n",dd,ndir,nchan,nrow,l,m);
    //printf("l,m: %f %f %f\n",l,m,n);
    //printf("*UVWin %f %f %f\n",l,m,n);
    VisIn=p0;
    UVWin=p1;
    for ( i=0; i<nrow; i++)  {
	phase=(*UVWin++)*l;
	//printf("cc %f \n",phase);
	phase+=(*UVWin++)*m;
	//printf("cc %f \n",phase);
	phase+=(*UVWin++)*n;
	//printf("cc %f \n",phase);
	for(ch=0;ch<nchan;ch++){ 
	  //printf("ch: %i %f\n",ch,WaveL[ch]);
	  c1=c0/WaveL[ch];
	  result=Flux[dd]*cexp(phase*c1);
	  //printf("cc %f %f %f %fi\n",phase,Flux[dd],creal(result),cimag(result));
	  //printf("\n");
	  *VisIn++   += result;
	  VisIn++;VisIn++;
	  *VisIn++   += result;
	}
	
    }
  }
  
  //free_Carrayptrs2(UVWin);
  //return Py_BuildValue("OO", NpVisInRe,NpVisInIm);//NpUVWin);
  return PyArray_Return(NpVisIn);
}




static PyObject *predictParallel(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyArrayObject *NpVisIn, *NpUVWin, *NpLMin, *matout,*NpLM, *NpWaveL,*NpFlux,*NpRowChunks;
  double *LM;
  int nrow,npol,nsources,dim[2];
  
  if (!PyArg_ParseTuple(args, "OO!O!O!O!O!", &ObjVisIn, &PyArray_Type, &NpUVWin, 
			&PyArray_Type, &NpLM, &PyArray_Type, &NpWaveL, &PyArray_Type, &NpFlux,
			&PyArray_Type, &NpRowChunks
			))  return NULL;
  
  NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX128, 0, 3);
  
  //printf("*UVWin +1j\n");
  LM=D_ptr(NpLM);
  
  /* printf("\n"); */
  /* printf("%i %i\n",RowChunks[0],RowChunks[1]); */


  int dd,nchan,ndir,nchunk;
  nrow=NpVisIn->dimensions[0];
  nchan=NpVisIn->dimensions[1];
  nchunk=NpRowChunks->dimensions[0];

  ndir=NpLM->dimensions[0];

  double l,m,n;

/* { */
/*   #pragma omp parallel */
/*     printf("Hello, world.\n"); */
/* } */

    
  for(dd=0;dd<ndir;dd++){
    l=LM[dd];
    m=LM[dd];
    n=sqrt(1.-l*l-m*m)-1.;
    printf("dd: %i/%i nchan=%i nrow=%i (l,m)=(%f,%f)\n",dd,ndir,nchan,nrow,l,m);
    int ichunk;
    {
      //#pragma omp parallel// for private(ichunk,l,m,n)
#pragma omp parallel for
    for (ichunk = 0; ichunk < nchunk-1; ichunk++){
      int *RowChunks;
      int istart,iend,irow,ch;
      double complex  *VisIn;
      double phase;
      double *UVWin;
      double complex result;
      double complex c0,c1;
      double *WaveL,*Flux;

      WaveL=D_ptr(NpWaveL);
      Flux=D_ptr(NpFlux);
      c0=2.*3.141592*I;
      RowChunks=I_ptr(NpRowChunks);

      //printf("%i->%i\n",RowChunks[0],RowChunks[1]);
      istart=RowChunks[ichunk];
      iend=RowChunks[ichunk+1];

      VisIn=DC_ptr(NpVisIn)+istart*nchan*4;
      UVWin=D_ptr(NpUVWin)+istart*3;
      

      //printf("%i->%i %i/%i\n",istart,iend,ichunk,nchunk);
      for ( irow=istart; irow<iend; irow++)  {
      	phase=(*UVWin++)*l;
      	phase+=(*UVWin++)*m;
      	phase+=(*UVWin++)*n;
      	for(ch=0;ch<nchan;ch++){
      	  c1=c0/WaveL[ch];
	  //printf("c1 %f %f %f\n",creal(c1),cimag(c1),WaveL[ch]);
      	  result=Flux[dd]*cexp(phase*c1);
	  //printf("result %f %f\n",creal(result),cimag(result));
      	  *VisIn++   += result;
      	  VisIn++;VisIn++;
      	  *VisIn++   += result;
      	}
      }
    }
    }
  }
  return PyArray_Return(NpVisIn);
}



static PyObject *predictParallelReverse(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyArrayObject *NpVisIn, *NpUVWin, *NpLMin, *matout,*NpLM, *NpWaveL,*NpFlux,*NpRowChunks;
  double *LM;
  int nrow,npol,nsources,dim[2];
  
  if (!PyArg_ParseTuple(args, "OO!O!O!O!O!", &ObjVisIn, &PyArray_Type, &NpUVWin, 
			&PyArray_Type, &NpLM, &PyArray_Type, &NpWaveL, &PyArray_Type, &NpFlux,
			&PyArray_Type, &NpRowChunks
			))  return NULL;
  
  NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX128, 0, 3);
  
  //printf("*UVWin +1j\n");
  LM=D_ptr(NpLM);
  
  /* printf("\n"); */
  /* printf("%i %i\n",RowChunks[0],RowChunks[1]); */


  int dd,nchan,ndir,nchunk;
  nrow=NpVisIn->dimensions[1];
  nchan=NpVisIn->dimensions[0];
  nchunk=NpRowChunks->dimensions[0];

  ndir=NpLM->dimensions[0];

  double l,m,n;

/* { */
/*   #pragma omp parallel */
/*     printf("Hello, world.\n"); */
/* } */

    
  for(dd=0;dd<ndir;dd++){
    l=LM[dd];
    m=LM[dd];
    n=sqrt(1.-l*l-m*m)-1.;
    printf("dd: %i/%i nchan=%i nrow=%i (l,m)=(%f,%f)\n",dd,ndir,nchan,nrow,l,m);
    int ichunk;
    double complex c0,c1;
    double *WaveL;
    WaveL=D_ptr(NpWaveL);
    c0=2.*3.141592*I;
    int ch;
    for(ch=0;ch<nchan;ch++){
      c1=c0/WaveL[ch];
      {
      //#pragma omp parallel// for private(ichunk,l,m,n)
#pragma omp parallel for
	for (ichunk = 0; ichunk < nchunk-1; ichunk++){
	  int *RowChunks;
	  int istart,iend,irow;
	  double complex  *VisIn;
	  double phase;
	  double *UVWin;
	  double complex result;
	  double *Flux;
	  
	  Flux=D_ptr(NpFlux);
	  RowChunks=I_ptr(NpRowChunks);
	  
	  //printf("%i->%i\n",RowChunks[0],RowChunks[1]);
	  istart=RowChunks[ichunk];
	  iend=RowChunks[ichunk+1];
	  
	  VisIn=DC_ptr(NpVisIn)+istart*4+ch*nrow*4;
	  UVWin=D_ptr(NpUVWin)+istart*3;
	  

      //printf("%i->%i %i/%i\n",istart,iend,ichunk,nchunk);
	  for ( irow=istart; irow<iend; irow++)  {
	    phase=(*UVWin++)*l;
	    phase+=(*UVWin++)*m;
	    phase+=(*UVWin++)*n;
	    //printf("c1 %f %f %f\n",creal(c1),cimag(c1),WaveL[ch]);
	    result=Flux[dd]*cexp(phase*c1);
	    //printf("result %f %f\n",creal(result),cimag(result));
	    *VisIn++   += result;
	    VisIn++;VisIn++;
	    *VisIn++   += result;
	  }
	}
      }
    }
  }
  return PyArray_Return(NpVisIn);
}


static PyObject *DotParallel(PyObject *self, PyObject *args)
{
  PyArrayObject *NpA, *NpB, *NpOut, *NpRowChunks;
  
  if (!PyArg_ParseTuple(args, "O!O!O!",
			&PyArray_Type, &NpA,
			&PyArray_Type, &NpB,
			&PyArray_Type, &NpRowChunks
			))  return NULL;
  

  
  int nchunk,ichunk,L,M,N;
  L=NpA->dimensions[0];
  M=NpA->dimensions[1];
  N=NpB->dimensions[1];

  npy_intp wt_dims[2] = {L,N};
  NpOut = (PyArrayObject *) PyArray_SimpleNew(2, wt_dims, NPY_FLOAT64);


  nchunk=NpRowChunks->dimensions[0];
  
#pragma omp parallel for
  for (ichunk = 0; ichunk < nchunk-1; ichunk++){
    int *RowChunks;
    int istart,iend,irow;
    double  *a, *b, *out;
    double  *a0, *b0, *out0;
    int i,j,k;
    
    a0=D_ptr(NpA);
    b0=D_ptr(NpB);
    out0=D_ptr(NpOut);
    RowChunks=I_ptr(NpRowChunks);
    
    //printf("%i->%i\n",RowChunks[0],RowChunks[1]);
    istart=RowChunks[ichunk];
    iend=RowChunks[ichunk+1];
    
    //    printf("%i->%i %i/%i\n",istart,iend,ichunk,nchunk-1);
    for ( i=istart; i<iend; i++)  {
      for ( j=0; j<N; j++)  {
	a=a0+i*M;
	b=b0+j*N;
	out=out0+i*N+j;
	*out=0.;
	for ( k=0; k<M; k++)  {
	  *out+=(*a++)*(*b++);
	  //printf("(%i,%i)[%i]    %6.3f * %6.3f = %6.3f\n",i,j,k,*a,*b,*out);
	  //a++;
	  //b++;
	}
	//printf("\n");
	
      }
    }
  }

  return PyArray_Return(NpOut);
}


static PyObject *predictSols(PyObject *self, PyObject *args)
{
  PyObject *ObjVisIn;
  PyArrayObject *NpVisIn, *NpUVWin, *NpLMin, *matout,*NpLM,
    *NpWaveL,*NpSols,*NpTimesSols,*NpA0,*NpA1,*NpInfo;
  int *A0,*A1;
  double *UVWin,*LM,*WaveL,*TimesSols,*Times, *Info;
  double complex  *VisIn, *Sols;
  int nrow,npol,nsources,i,dim[2];

  
  if (!PyArg_ParseTuple(args, "OO!O!O!O!O!O!O!", 
			&ObjVisIn, 
			&PyArray_Type, &NpA0, 
			&PyArray_Type, &NpA1, 
			&PyArray_Type, &NpUVWin, 
			&PyArray_Type, &NpLM,
			&PyArray_Type, &NpWaveL, 
			&PyArray_Type, &NpSols,
			&PyArray_Type, &NpInfo))  return NULL;
  
  NpVisIn = (PyArrayObject *) PyArray_ContiguousFromObject(ObjVisIn, PyArray_COMPLEX128, 0, 3);
  //NpUVWin = (PyArrayObject *) PyArray_ContiguousFromObject(ObjUVWin, PyArray_DOUBLE, 0, 3);
  //NpLM = (PyArrayObject *) PyArray_ContiguousFromObject(ObjLM, PyArray_DOUBLE, 0, 3);
  
  //  printf("*UVWin +1j\n");
  VisIn = DC_ptr(NpVisIn);
  Sols  = DC_ptr(NpSols);
  Info  = D_ptr(NpInfo);
  double RefWave=Info[0];

  A0=I_ptr(NpA0);
  A1=I_ptr(NpA1);

  UVWin = D_ptr(NpUVWin);
  LM    = D_ptr(NpLM);
  WaveL = D_ptr(NpWaveL);
  
  int ch,dd,nchan,ndir,na;
  nrow=NpVisIn->dimensions[0];
  nchan=NpVisIn->dimensions[1];
  na=NpSols->dimensions[2];


  int nlm;
  ndir=NpLM->dimensions[0];
  nlm=NpLM->dimensions[1];

  //  printf("NpLM dimensions = (%i, %i)\n",ndir,nlm);


  /* Get the dimensions of the input */
  
  /* Make a new double matrix of same dims */
  //matout=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
  
  
  /* Do the calculation. */
  double phase,l,m,n,u,v,w;
  double complex c0,c1,result;
  c0=2.*3.141592*I;
  double complex *p0;
  double *p1;
  p0=VisIn;
  p1=UVWin;
  
  double complex *J0, *J1;
  double complex JJ[4];

  //  printf("nd=%i, Nf=%i, na=%i\n", ndir,nchan,na);
  /* //Sols shape: Nd, Nf, Na, 4 */
  /* J0=Sols+ 1*na*nchan*4 + 2*na*4 + 3*4; */
  /* J1=Sols+ 0*na*nchan*4 + 6*na*4 + 8*4; */
  /* Prod22H(J0,J1,JJ); */
  /* int pol; */
  /* for(pol=0; pol<4; pol++){ */
  /*   printf("J0[%i] = %f + i%f\n", pol, creal(J0[pol]), cimag(J0[pol])); */
  /* } */
  /* printf("\n"); */
  /* for(pol=0; pol<4; pol++){ */
  /*   printf("J1[%i] = %f + i%f\n", pol, creal(J1[pol]), cimag(J1[pol])); */
  /* } */
  /* printf("\n"); */
  /* for(pol=0; pol<4; pol++){ */
  /*   printf("JJ[%i] = %f + i%f\n", pol, creal(JJ[pol]), cimag(JJ[pol])); */
  /* } */
  /* assert(1==0); */


  /* for(dd=0;dd<6;dd++){ */
  /*   printf("dd: %i , %f\n",dd,LM[dd]); */
  /* } */

  double ThisFlux, Flux, Alpha;

  for(dd=0;dd<ndir;dd++){
    l=LM[0];
    m=LM[1];
    Flux=LM[2];
    Alpha=LM[3];
    LM+=nlm;
    n=sqrt(1.-l*l-m*m)-1.;
    //printf("====================================\n");
    //printf("dd: %i/%i nchan=%i nrow=%i (l,m,s,al)=(%f,%f,%f,%f)\n",dd,ndir,nchan,nrow,l,m,Flux,Alpha);
    //printf("l,m: %f %f %f\n",l,m,n);
    //printf("*UVWin %f %f %f\n",l,m,n);
    VisIn=p0;
    UVWin=p1;
    for ( i=0; i<nrow; i++)  {
	phase=(*UVWin++)*l;
	//printf("cc %f \n",phase);
	phase+=(*UVWin++)*m;
	//printf("cc %f \n",phase);
	phase+=(*UVWin++)*n;
	//printf("cc %f \n",phase);
	for(ch=0;ch<nchan;ch++){ 
	  //printf("ch: %i %f\n",ch,WaveL[ch]);
	  c1=c0/WaveL[ch];
	  ThisFlux=Flux*pow(RefWave/WaveL[ch],Alpha);
	  result=ThisFlux*cexp(phase*c1);
	  
	  //Sols shape: Nd, Nf, Na, 4
	  J0=Sols+ dd*na*nchan*4 + ch*na*4 + A0[i]*4;
	  J1=Sols+ dd*na*nchan*4 + ch*na*4 + A1[i]*4;

	  

	  Prod22H(J0,J1,JJ);

	  /* int pol; */
	  /* for(pol=0; pol<4; pol++){ */
	  /*   printf("J0[%i] = %f + i%f\n", pol, creal(J0[pol]), cimag(J0[pol])); */
	  /* } */
	  /* printf("\n"); */
	  /* for(pol=0; pol<4; pol++){ */
	  /*   printf("J1[%i] = %f + i%f\n", pol, creal(J1[pol]), cimag(J1[pol])); */
	  /* } */
	  /* printf("\n"); */
	  /* for(pol=0; pol<4; pol++){ */
	  /*   printf("JJ[%i] = %f + i%f\n", pol, creal(JJ[pol]), cimag(JJ[pol])); */
	  /* } */
  
	  /* printf("=============\n"); */
	  /* printf("dd: %i/%i nchan=%i nrow=%i (l,m,s,al)=(%f,%f,%f,%f)\n",dd,ndir,nchan,nrow,l,m,Flux,Alpha); */
 	  /* printf("cc %f %f %f %fi (abs=%f)\n",phase,ThisFlux,creal(result),cimag(result),cabs(result)); */
	  /* //printf("\n"); */
	  

	  VisIn[0]+=JJ[0]*result;
	  VisIn[1]+=JJ[1]*result;
	  VisIn[2]+=JJ[2]*result;
	  VisIn[3]+=JJ[3]*result;

	  /* VisIn[0]+=result; */
	  /* VisIn[3]+=result; */

	  VisIn+=4;

	  //if(i>10){assert(1==0);}

	}
	
    }
  }
  //assert(1==0);
  //free_Carrayptrs2(UVWin);
  //return Py_BuildValue("OO", NpVisInRe,NpVisInIm);//NpUVWin);
  return PyArray_Return(NpVisIn);
}



static PyObject *BatchProd(PyObject *self, PyObject *args)
{


  PyArrayObject *NpA0,*NpA1,*NpOut;

  int nrow,npol,nsources,i,dim[2];

  
  if (!PyArg_ParseTuple(args, "O!O!", 
			&PyArray_Type, &NpA0,
			&PyArray_Type, &NpA1))  return NULL;
  
  int ndir, nchan, na;
  ndir=NpA0->dimensions[0];
  nchan=NpA0->dimensions[1];
  na=NpA0->dimensions[2];
  
  //  printf("i");

  npy_intp wt_dims[4] = {ndir, nchan, na, NpA0->dimensions[3]};
  NpOut = (PyArrayObject *) PyArray_SimpleNew(4, wt_dims, NPY_COMPLEX128);
  

  double complex *A0, *A1, *Out;
  double complex *a0, *a1, *out;
  A0=DC_ptr(NpA0);
  A1=DC_ptr(NpA1);
  Out=DC_ptr(NpOut);
  

  int ch, dir, ant, pos;
  for(dir=0; dir<ndir; dir++){
    for(ch=0; ch<nchan; ch++){
      for(ant=0; ant<na; ant++){
  	pos=dir*nchan*na*4+ch*na*4+ant*4;
  	a0=A0+pos;
  	a1=A1+pos;
  	out=Out+pos;
  	Prod22(a0,a1,out);
	/* printf("%i %i %i\n",dir,ch,ant); */
	/* printf("pos = %i\n", pos); */
	  
	/* int pol; */
	/* for(pol=0; pol<4; pol++){ */
	/*   printf("A0[%i] = %f + i%f\n", pol, creal(a0[pol]), cimag(a0[pol])); */
	/* } */
	/* printf("\n"); */
	/* for(pol=0; pol<4; pol++){ */
	/*   printf("a1[%i] = %f + i%f\n", pol, creal(a1[pol]), cimag(a1[pol])); */
	/* } */
	/* printf("\n"); */
	/* for(pol=0; pol<4; pol++){ */
	/*   printf("ou[%i] = %f + i%f\n", pol, creal(out[pol]), cimag(out[pol])); */
	/* } */

      }
    }
  }


  return PyArray_Return(NpOut);
  //return PyArray_Return(NpA0);
}
