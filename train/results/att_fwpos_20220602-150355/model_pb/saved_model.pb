¨ó%
ÿã
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02unknown8Ä"

conv_att/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À *(
shared_nameconv_att/dense_5/kernel

+conv_att/dense_5/kernel/Read/ReadVariableOpReadVariableOpconv_att/dense_5/kernel*
_output_shapes
:	À *
dtype0

conv_att/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconv_att/dense_5/bias
{
)conv_att/dense_5/bias/Read/ReadVariableOpReadVariableOpconv_att/dense_5/bias*
_output_shapes
: *
dtype0

conv_att/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$ *(
shared_nameconv_att/dense_6/kernel

+conv_att/dense_6/kernel/Read/ReadVariableOpReadVariableOpconv_att/dense_6/kernel*
_output_shapes

:$ *
dtype0

conv_att/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconv_att/dense_6/bias
{
)conv_att/dense_6/bias/Read/ReadVariableOpReadVariableOpconv_att/dense_6/bias*
_output_shapes
: *
dtype0

conv_att/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	À *(
shared_nameconv_att/dense_7/kernel

+conv_att/dense_7/kernel/Read/ReadVariableOpReadVariableOpconv_att/dense_7/kernel*
_output_shapes
:	À *
dtype0

conv_att/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconv_att/dense_7/bias
{
)conv_att/dense_7/bias/Read/ReadVariableOpReadVariableOpconv_att/dense_7/bias*
_output_shapes
: *
dtype0

conv_att/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *(
shared_nameconv_att/dense_8/kernel

+conv_att/dense_8/kernel/Read/ReadVariableOpReadVariableOpconv_att/dense_8/kernel*
_output_shapes

:@ *
dtype0

conv_att/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconv_att/dense_8/bias
{
)conv_att/dense_8/bias/Read/ReadVariableOpReadVariableOpconv_att/dense_8/bias*
_output_shapes
: *
dtype0

conv_att/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *(
shared_nameconv_att/dense_9/kernel

+conv_att/dense_9/kernel/Read/ReadVariableOpReadVariableOpconv_att/dense_9/kernel*
_output_shapes

: *
dtype0

conv_att/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv_att/dense_9/bias
{
)conv_att/dense_9/bias/Read/ReadVariableOpReadVariableOpconv_att/dense_9/bias*
_output_shapes
:*
dtype0

conv_att/enc_past/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!conv_att/enc_past/conv1d/kernel

3conv_att/enc_past/conv1d/kernel/Read/ReadVariableOpReadVariableOpconv_att/enc_past/conv1d/kernel*"
_output_shapes
: *
dtype0

conv_att/enc_past/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameconv_att/enc_past/conv1d/bias

1conv_att/enc_past/conv1d/bias/Read/ReadVariableOpReadVariableOpconv_att/enc_past/conv1d/bias*
_output_shapes
: *
dtype0
¢
!conv_att/enc_past/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *2
shared_name#!conv_att/enc_past/conv1d_1/kernel

5conv_att/enc_past/conv1d_1/kernel/Read/ReadVariableOpReadVariableOp!conv_att/enc_past/conv1d_1/kernel*"
_output_shapes
:  *
dtype0

conv_att/enc_past/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!conv_att/enc_past/conv1d_1/bias

3conv_att/enc_past/conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv_att/enc_past/conv1d_1/bias*
_output_shapes
: *
dtype0
¢
!conv_att/enc_past/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *2
shared_name#!conv_att/enc_past/conv1d_2/kernel

5conv_att/enc_past/conv1d_2/kernel/Read/ReadVariableOpReadVariableOp!conv_att/enc_past/conv1d_2/kernel*"
_output_shapes
:  *
dtype0

conv_att/enc_past/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!conv_att/enc_past/conv1d_2/bias

3conv_att/enc_past/conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv_att/enc_past/conv1d_2/bias*
_output_shapes
: *
dtype0

!conv_att/spatial_att/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!conv_att/spatial_att/dense/kernel

5conv_att/spatial_att/dense/kernel/Read/ReadVariableOpReadVariableOp!conv_att/spatial_att/dense/kernel*
_output_shapes

:*
dtype0

conv_att/spatial_att/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!conv_att/spatial_att/dense/bias

3conv_att/spatial_att/dense/bias/Read/ReadVariableOpReadVariableOpconv_att/spatial_att/dense/bias*
_output_shapes
:*
dtype0
¢
#conv_att/spatial_att/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#conv_att/spatial_att/dense_1/kernel

7conv_att/spatial_att/dense_1/kernel/Read/ReadVariableOpReadVariableOp#conv_att/spatial_att/dense_1/kernel*
_output_shapes

:*
dtype0

!conv_att/spatial_att/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!conv_att/spatial_att/dense_1/bias

5conv_att/spatial_att/dense_1/bias/Read/ReadVariableOpReadVariableOp!conv_att/spatial_att/dense_1/bias*
_output_shapes
:*
dtype0
¢
#conv_att/spatial_att/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#conv_att/spatial_att/dense_2/kernel

7conv_att/spatial_att/dense_2/kernel/Read/ReadVariableOpReadVariableOp#conv_att/spatial_att/dense_2/kernel*
_output_shapes

: *
dtype0
¢
#conv_att/spatial_att/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#conv_att/spatial_att/dense_3/kernel

7conv_att/spatial_att/dense_3/kernel/Read/ReadVariableOpReadVariableOp#conv_att/spatial_att/dense_3/kernel*
_output_shapes

:*
dtype0
¢
#conv_att/spatial_att/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#conv_att/spatial_att/dense_4/kernel

7conv_att/spatial_att/dense_4/kernel/Read/ReadVariableOpReadVariableOp#conv_att/spatial_att/dense_4/kernel*
_output_shapes

:*
dtype0
¨
$conv_att/enc_forward/conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$conv_att/enc_forward/conv1d_3/kernel
¡
8conv_att/enc_forward/conv1d_3/kernel/Read/ReadVariableOpReadVariableOp$conv_att/enc_forward/conv1d_3/kernel*"
_output_shapes
: *
dtype0

"conv_att/enc_forward/conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"conv_att/enc_forward/conv1d_3/bias

6conv_att/enc_forward/conv1d_3/bias/Read/ReadVariableOpReadVariableOp"conv_att/enc_forward/conv1d_3/bias*
_output_shapes
: *
dtype0
¨
$conv_att/enc_forward/conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *5
shared_name&$conv_att/enc_forward/conv1d_4/kernel
¡
8conv_att/enc_forward/conv1d_4/kernel/Read/ReadVariableOpReadVariableOp$conv_att/enc_forward/conv1d_4/kernel*"
_output_shapes
:  *
dtype0

"conv_att/enc_forward/conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"conv_att/enc_forward/conv1d_4/bias

6conv_att/enc_forward/conv1d_4/bias/Read/ReadVariableOpReadVariableOp"conv_att/enc_forward/conv1d_4/bias*
_output_shapes
: *
dtype0
¨
$conv_att/enc_forward/conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *5
shared_name&$conv_att/enc_forward/conv1d_5/kernel
¡
8conv_att/enc_forward/conv1d_5/kernel/Read/ReadVariableOpReadVariableOp$conv_att/enc_forward/conv1d_5/kernel*"
_output_shapes
:  *
dtype0

"conv_att/enc_forward/conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"conv_att/enc_forward/conv1d_5/bias

6conv_att/enc_forward/conv1d_5/bias/Read/ReadVariableOpReadVariableOp"conv_att/enc_forward/conv1d_5/bias*
_output_shapes
: *
dtype0

NoOpNoOp
½
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*÷
valueìBè Bà
Ê
enc_past
spatial_att
enc_forward

dense1

dense2

dense3

dense4

dense5
	relu

tanh
flatten
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
Ê
filters
	sizes
strides
	convs
relu
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
ä

dense1

 dense2
!w_e
"u_e
#v_e
$relu
%tanh
&softmax
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*
Ê
-filters
	.sizes
/strides
	0convs
1relu
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses*
¦

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
¦

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*
¦

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses*
¦

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses*
¦

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses*

`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses* 

f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses* 

l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
ç
r0
s1
t2
u3
v4
w5
x6
y7
z8
{9
|10
}11
~12
13
14
15
16
17
18
819
920
@21
A22
H23
I24
P25
Q26
X27
Y28*
ç
r0
s1
t2
u3
v4
w5
x6
y7
z8
{9
|10
}11
~12
13
14
15
16
17
18
819
920
@21
A22
H23
I24
P25
Q26
X27
Y28*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
* 
* 
* 

0
1
2*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
.
r0
s1
t2
u3
v4
w5*
.
r0
s1
t2
u3
v4
w5*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
¬

xkernel
ybias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

zkernel
{bias
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£__call__
+¤&call_and_return_all_conditional_losses*
¢

|kernel
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
©__call__
+ª&call_and_return_all_conditional_losses*
¢

}kernel
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯__call__
+°&call_and_return_all_conditional_losses*
¢

~kernel
±	variables
²trainable_variables
³regularization_losses
´	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses*

·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
»__call__
+¼&call_and_return_all_conditional_losses* 

½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses* 

Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses* 
5
x0
y1
z2
{3
|4
}5
~6*
5
x0
y1
z2
{3
|4
}5
~6*
* 

Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Î0
Ï1
Ð2*

Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses* 
3
0
1
2
3
4
5*
3
0
1
2
3
4
5*
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUEconv_att/dense_5/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_att/dense_5/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 

Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUEconv_att/dense_6/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_att/dense_6/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*
* 

ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUEconv_att/dense_7/kernel(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_att/dense_7/bias&dense3/bias/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*

H0
I1*
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUEconv_att/dense_8/kernel(dense4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_att/dense_8/bias&dense4/bias/.ATTRIBUTES/VARIABLE_VALUE*

P0
Q1*

P0
Q1*
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUEconv_att/dense_9/kernel(dense5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEconv_att/dense_9/bias&dense5/bias/.ATTRIBUTES/VARIABLE_VALUE*

X0
Y1*

X0
Y1*
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv_att/enc_past/conv1d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv_att/enc_past/conv1d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!conv_att/enc_past/conv1d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEconv_att/enc_past/conv1d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!conv_att/enc_past/conv1d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEconv_att/enc_past/conv1d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!conv_att/spatial_att/dense/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEconv_att/spatial_att/dense/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#conv_att/spatial_att/dense_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!conv_att/spatial_att/dense_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#conv_att/spatial_att/dense_2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#conv_att/spatial_att/dense_3/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#conv_att/spatial_att/dense_4/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$conv_att/enc_forward/conv1d_3/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"conv_att/enc_forward/conv1d_3/bias'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$conv_att/enc_forward/conv1d_4/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"conv_att/enc_forward/conv1d_4/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$conv_att/enc_forward/conv1d_5/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"conv_att/enc_forward/conv1d_5/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*
* 
* 
* 
* 
¬

rkernel
sbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

tkernel
ubias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

vkernel
wbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
#
0
1
2
3*
* 
* 
* 

x0
y1*

x0
y1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

z0
{1*

z0
{1*
* 

 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
	variables
 trainable_variables
¡regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses*
* 
* 

|0*

|0*
* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
¥	variables
¦trainable_variables
§regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses*
* 
* 

}0*

}0*
* 

ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
«	variables
¬trainable_variables
­regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses*
* 
* 

~0*

~0*
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
±	variables
²trainable_variables
³regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
·	variables
¸trainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses* 
* 
* 
* 
<
0
 1
!2
"3
#4
$5
%6
&7*
* 
* 
* 
­

kernel
	bias
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses*
®
kernel
	bias
É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses*
®
kernel
	bias
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses*
* 
* 
* 

Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
Ñ	variables
Òtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses* 
* 
* 
* 
#
Î0
Ï1
Ð2
13*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

r0
s1*

r0
s1*
* 

Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

t0
u1*

t0
u1*
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

v0
w1*

v0
w1*
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
É	variables
Êtrainable_variables
Ëregularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

serving_default_args_0Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_args_1Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿd

serving_default_args_2Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ


serving_default_args_3Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿd
Õ

StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_1serving_default_args_2serving_default_args_3conv_att/enc_past/conv1d/kernelconv_att/enc_past/conv1d/bias!conv_att/enc_past/conv1d_1/kernelconv_att/enc_past/conv1d_1/bias!conv_att/enc_past/conv1d_2/kernelconv_att/enc_past/conv1d_2/biasconv_att/dense_5/kernelconv_att/dense_5/biasconv_att/dense_6/kernelconv_att/dense_6/bias!conv_att/spatial_att/dense/kernelconv_att/spatial_att/dense/bias#conv_att/spatial_att/dense_1/kernel!conv_att/spatial_att/dense_1/bias#conv_att/spatial_att/dense_3/kernel#conv_att/spatial_att/dense_2/kernel#conv_att/spatial_att/dense_4/kernel$conv_att/enc_forward/conv1d_3/kernel"conv_att/enc_forward/conv1d_3/bias$conv_att/enc_forward/conv1d_4/kernel"conv_att/enc_forward/conv1d_4/bias$conv_att/enc_forward/conv1d_5/kernel"conv_att/enc_forward/conv1d_5/biasconv_att/dense_7/kernelconv_att/dense_7/biasconv_att/dense_8/kernelconv_att/dense_8/biasconv_att/dense_9/kernelconv_att/dense_9/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*?
_read_only_resource_inputs!
	
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *.
f)R'
%__inference_signature_wrapper_1631157
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+conv_att/dense_5/kernel/Read/ReadVariableOp)conv_att/dense_5/bias/Read/ReadVariableOp+conv_att/dense_6/kernel/Read/ReadVariableOp)conv_att/dense_6/bias/Read/ReadVariableOp+conv_att/dense_7/kernel/Read/ReadVariableOp)conv_att/dense_7/bias/Read/ReadVariableOp+conv_att/dense_8/kernel/Read/ReadVariableOp)conv_att/dense_8/bias/Read/ReadVariableOp+conv_att/dense_9/kernel/Read/ReadVariableOp)conv_att/dense_9/bias/Read/ReadVariableOp3conv_att/enc_past/conv1d/kernel/Read/ReadVariableOp1conv_att/enc_past/conv1d/bias/Read/ReadVariableOp5conv_att/enc_past/conv1d_1/kernel/Read/ReadVariableOp3conv_att/enc_past/conv1d_1/bias/Read/ReadVariableOp5conv_att/enc_past/conv1d_2/kernel/Read/ReadVariableOp3conv_att/enc_past/conv1d_2/bias/Read/ReadVariableOp5conv_att/spatial_att/dense/kernel/Read/ReadVariableOp3conv_att/spatial_att/dense/bias/Read/ReadVariableOp7conv_att/spatial_att/dense_1/kernel/Read/ReadVariableOp5conv_att/spatial_att/dense_1/bias/Read/ReadVariableOp7conv_att/spatial_att/dense_2/kernel/Read/ReadVariableOp7conv_att/spatial_att/dense_3/kernel/Read/ReadVariableOp7conv_att/spatial_att/dense_4/kernel/Read/ReadVariableOp8conv_att/enc_forward/conv1d_3/kernel/Read/ReadVariableOp6conv_att/enc_forward/conv1d_3/bias/Read/ReadVariableOp8conv_att/enc_forward/conv1d_4/kernel/Read/ReadVariableOp6conv_att/enc_forward/conv1d_4/bias/Read/ReadVariableOp8conv_att/enc_forward/conv1d_5/kernel/Read/ReadVariableOp6conv_att/enc_forward/conv1d_5/bias/Read/ReadVariableOpConst**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *)
f$R"
 __inference__traced_save_1632424
Î	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_att/dense_5/kernelconv_att/dense_5/biasconv_att/dense_6/kernelconv_att/dense_6/biasconv_att/dense_7/kernelconv_att/dense_7/biasconv_att/dense_8/kernelconv_att/dense_8/biasconv_att/dense_9/kernelconv_att/dense_9/biasconv_att/enc_past/conv1d/kernelconv_att/enc_past/conv1d/bias!conv_att/enc_past/conv1d_1/kernelconv_att/enc_past/conv1d_1/bias!conv_att/enc_past/conv1d_2/kernelconv_att/enc_past/conv1d_2/bias!conv_att/spatial_att/dense/kernelconv_att/spatial_att/dense/bias#conv_att/spatial_att/dense_1/kernel!conv_att/spatial_att/dense_1/bias#conv_att/spatial_att/dense_2/kernel#conv_att/spatial_att/dense_3/kernel#conv_att/spatial_att/dense_4/kernel$conv_att/enc_forward/conv1d_3/kernel"conv_att/enc_forward/conv1d_3/bias$conv_att/enc_forward/conv1d_4/kernel"conv_att/enc_forward/conv1d_4/bias$conv_att/enc_forward/conv1d_5/kernel"conv_att/enc_forward/conv1d_5/bias*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *,
f'R%
#__inference__traced_restore_1632521©È 
Û

)__inference_dense_6_layer_call_fn_1631853

inputs
unknown:$ 
	unknown_0: 
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1629342s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
$: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
 
_user_specified_nameinputs
Ä

³
-__inference_spatial_att_layer_call_fn_1631311	
query
feature
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4: 
	unknown_5:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallqueryfeatureunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*)
_read_only_resource_inputs
	*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_spatial_att_layer_call_and_return_conditional_losses_1629910s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
 :ÿÿÿÿÿÿÿÿÿd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 

_user_specified_namequery:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	feature
	

-__inference_enc_forward_layer_call_fn_1631596
x
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629067w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ
d: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d

_user_specified_namex
Þ

)__inference_dense_7_layer_call_fn_1631892

inputs
unknown:	À 
	unknown_0: 
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1629554s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
À: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
 
_user_specified_nameinputs
ù@

 __inference__traced_save_1632424
file_prefix6
2savev2_conv_att_dense_5_kernel_read_readvariableop4
0savev2_conv_att_dense_5_bias_read_readvariableop6
2savev2_conv_att_dense_6_kernel_read_readvariableop4
0savev2_conv_att_dense_6_bias_read_readvariableop6
2savev2_conv_att_dense_7_kernel_read_readvariableop4
0savev2_conv_att_dense_7_bias_read_readvariableop6
2savev2_conv_att_dense_8_kernel_read_readvariableop4
0savev2_conv_att_dense_8_bias_read_readvariableop6
2savev2_conv_att_dense_9_kernel_read_readvariableop4
0savev2_conv_att_dense_9_bias_read_readvariableop>
:savev2_conv_att_enc_past_conv1d_kernel_read_readvariableop<
8savev2_conv_att_enc_past_conv1d_bias_read_readvariableop@
<savev2_conv_att_enc_past_conv1d_1_kernel_read_readvariableop>
:savev2_conv_att_enc_past_conv1d_1_bias_read_readvariableop@
<savev2_conv_att_enc_past_conv1d_2_kernel_read_readvariableop>
:savev2_conv_att_enc_past_conv1d_2_bias_read_readvariableop@
<savev2_conv_att_spatial_att_dense_kernel_read_readvariableop>
:savev2_conv_att_spatial_att_dense_bias_read_readvariableopB
>savev2_conv_att_spatial_att_dense_1_kernel_read_readvariableop@
<savev2_conv_att_spatial_att_dense_1_bias_read_readvariableopB
>savev2_conv_att_spatial_att_dense_2_kernel_read_readvariableopB
>savev2_conv_att_spatial_att_dense_3_kernel_read_readvariableopB
>savev2_conv_att_spatial_att_dense_4_kernel_read_readvariableopC
?savev2_conv_att_enc_forward_conv1d_3_kernel_read_readvariableopA
=savev2_conv_att_enc_forward_conv1d_3_bias_read_readvariableopC
?savev2_conv_att_enc_forward_conv1d_4_kernel_read_readvariableopA
=savev2_conv_att_enc_forward_conv1d_4_bias_read_readvariableopC
?savev2_conv_att_enc_forward_conv1d_5_kernel_read_readvariableopA
=savev2_conv_att_enc_forward_conv1d_5_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¥

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Î	
valueÄ	BÁ	B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense3/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense4/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense4/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense5/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH©
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B í
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_conv_att_dense_5_kernel_read_readvariableop0savev2_conv_att_dense_5_bias_read_readvariableop2savev2_conv_att_dense_6_kernel_read_readvariableop0savev2_conv_att_dense_6_bias_read_readvariableop2savev2_conv_att_dense_7_kernel_read_readvariableop0savev2_conv_att_dense_7_bias_read_readvariableop2savev2_conv_att_dense_8_kernel_read_readvariableop0savev2_conv_att_dense_8_bias_read_readvariableop2savev2_conv_att_dense_9_kernel_read_readvariableop0savev2_conv_att_dense_9_bias_read_readvariableop:savev2_conv_att_enc_past_conv1d_kernel_read_readvariableop8savev2_conv_att_enc_past_conv1d_bias_read_readvariableop<savev2_conv_att_enc_past_conv1d_1_kernel_read_readvariableop:savev2_conv_att_enc_past_conv1d_1_bias_read_readvariableop<savev2_conv_att_enc_past_conv1d_2_kernel_read_readvariableop:savev2_conv_att_enc_past_conv1d_2_bias_read_readvariableop<savev2_conv_att_spatial_att_dense_kernel_read_readvariableop:savev2_conv_att_spatial_att_dense_bias_read_readvariableop>savev2_conv_att_spatial_att_dense_1_kernel_read_readvariableop<savev2_conv_att_spatial_att_dense_1_bias_read_readvariableop>savev2_conv_att_spatial_att_dense_2_kernel_read_readvariableop>savev2_conv_att_spatial_att_dense_3_kernel_read_readvariableop>savev2_conv_att_spatial_att_dense_4_kernel_read_readvariableop?savev2_conv_att_enc_forward_conv1d_3_kernel_read_readvariableop=savev2_conv_att_enc_forward_conv1d_3_bias_read_readvariableop?savev2_conv_att_enc_forward_conv1d_4_kernel_read_readvariableop=savev2_conv_att_enc_forward_conv1d_4_bias_read_readvariableop?savev2_conv_att_enc_forward_conv1d_5_kernel_read_readvariableop=savev2_conv_att_enc_forward_conv1d_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*¡
_input_shapes
: :	À : :$ : :	À : :@ : : :: : :  : :  : ::::: ::: : :  : :  : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	À : 

_output_shapes
: :$ 

_output_shapes

:$ : 

_output_shapes
: :%!

_output_shapes
:	À : 

_output_shapes
: :$ 

_output_shapes

:@ : 

_output_shapes
: :$	 

_output_shapes

: : 


_output_shapes
::($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: :$ 

_output_shapes

::$ 

_output_shapes

::($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :

_output_shapes
: 
Æ
E
)__inference_re_lu_2_layer_call_fn_1632076

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629064h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
^ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ 
 
_user_specified_nameinputs
Ë	
ö
D__inference_dense_5_layer_call_and_return_conditional_losses_1629293

inputs1
matmul_readvariableop_resource:	À -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	À *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Ã

E__inference_enc_past_layer_call_and_return_conditional_losses_1628733
x$
conv1d_1628666: 
conv1d_1628668: &
conv1d_1_1628694:  
conv1d_1_1628696: &
conv1d_2_1628721:  
conv1d_2_1628723: 
identity¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCalló
conv1d/StatefulPartitionedCallStatefulPartitionedCallxconv1d_1628666conv1d_1628668*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_1628665á
re_lu/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628676
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv1d_1_1628694conv1d_1_1628696*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1628693å
re_lu/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628703
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_1:output:0conv1d_2_1628721conv1d_2_1628723*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1628720å
re_lu/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628730s
IdentityIdentity re_lu/PartitionedCall_2:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ë
û
D__inference_dense_6_layer_call_and_return_conditional_losses_1631883

inputs3
!tensordot_readvariableop_resource:$ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:$ *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
 
_user_specified_nameinputs

Ð
H__inference_enc_forward_layer_call_and_return_conditional_losses_1631825
xJ
4conv1d_3_conv1d_expanddims_1_readvariableop_resource: I
;conv1d_3_squeeze_batch_dims_biasadd_readvariableop_resource: J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:  I
;conv1d_4_squeeze_batch_dims_biasadd_readvariableop_resource: J
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:  I
;conv1d_5_squeeze_batch_dims_biasadd_readvariableop_resource: 
identity¢+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp¢2conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp¢+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢2conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢2conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpi
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d_3/Conv1D/ExpandDims
ExpandDimsx'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
d¤
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: h
conv1d_3/Conv1D/ShapeShape#conv1d_3/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:m
#conv1d_3/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv1d_3/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv1d_3/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv1d_3/Conv1D/strided_sliceStridedSliceconv1d_3/Conv1D/Shape:output:0,conv1d_3/Conv1D/strided_slice/stack:output:0.conv1d_3/Conv1D/strided_slice/stack_1:output:0.conv1d_3/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv1d_3/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   d      ©
conv1d_3/Conv1D/ReshapeReshape#conv1d_3/Conv1D/ExpandDims:output:0&conv1d_3/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÌ
conv1d_3/Conv1D/Conv2DConv2D conv1d_3/Conv1D/Reshape:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb *
paddingVALID*
strides
t
conv1d_3/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   b       f
conv1d_3/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv1d_3/Conv1D/concatConcatV2&conv1d_3/Conv1D/strided_slice:output:0(conv1d_3/Conv1D/concat/values_1:output:0$conv1d_3/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv1d_3/Conv1D/Reshape_1Reshapeconv1d_3/Conv1D/Conv2D:output:0conv1d_3/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b  
conv1d_3/Conv1D/SqueezeSqueeze"conv1d_3/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *
squeeze_dims

ýÿÿÿÿÿÿÿÿq
!conv1d_3/squeeze_batch_dims/ShapeShape conv1d_3/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:y
/conv1d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv1d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ{
1conv1d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv1d_3/squeeze_batch_dims/strided_sliceStridedSlice*conv1d_3/squeeze_batch_dims/Shape:output:08conv1d_3/squeeze_batch_dims/strided_slice/stack:output:0:conv1d_3/squeeze_batch_dims/strided_slice/stack_1:output:0:conv1d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask~
)conv1d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿb       º
#conv1d_3/squeeze_batch_dims/ReshapeReshape conv1d_3/Conv1D/Squeeze:output:02conv1d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb ª
2conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv1d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Î
#conv1d_3/squeeze_batch_dims/BiasAddBiasAdd,conv1d_3/squeeze_batch_dims/Reshape:output:0:conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb |
+conv1d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"b       r
'conv1d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv1d_3/squeeze_batch_dims/concatConcatV22conv1d_3/squeeze_batch_dims/strided_slice:output:04conv1d_3/squeeze_batch_dims/concat/values_1:output:00conv1d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Å
%conv1d_3/squeeze_batch_dims/Reshape_1Reshape,conv1d_3/squeeze_batch_dims/BiasAdd:output:0+conv1d_3/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b ~
re_lu_2/ReluRelu.conv1d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b i
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ«
conv1d_4/Conv1D/ExpandDims
ExpandDimsre_lu_2/Relu:activations:0'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b ¤
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  h
conv1d_4/Conv1D/ShapeShape#conv1d_4/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:m
#conv1d_4/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv1d_4/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv1d_4/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv1d_4/Conv1D/strided_sliceStridedSliceconv1d_4/Conv1D/Shape:output:0,conv1d_4/Conv1D/strided_slice/stack:output:0.conv1d_4/Conv1D/strided_slice/stack_1:output:0.conv1d_4/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv1d_4/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   b       ©
conv1d_4/Conv1D/ReshapeReshape#conv1d_4/Conv1D/ExpandDims:output:0&conv1d_4/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb Ì
conv1d_4/Conv1D/Conv2DConv2D conv1d_4/Conv1D/Reshape:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` *
paddingVALID*
strides
t
conv1d_4/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   `       f
conv1d_4/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv1d_4/Conv1D/concatConcatV2&conv1d_4/Conv1D/strided_slice:output:0(conv1d_4/Conv1D/concat/values_1:output:0$conv1d_4/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv1d_4/Conv1D/Reshape_1Reshapeconv1d_4/Conv1D/Conv2D:output:0conv1d_4/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
`  
conv1d_4/Conv1D/SqueezeSqueeze"conv1d_4/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *
squeeze_dims

ýÿÿÿÿÿÿÿÿq
!conv1d_4/squeeze_batch_dims/ShapeShape conv1d_4/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:y
/conv1d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv1d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ{
1conv1d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv1d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv1d_4/squeeze_batch_dims/Shape:output:08conv1d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv1d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv1d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask~
)conv1d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ`       º
#conv1d_4/squeeze_batch_dims/ReshapeReshape conv1d_4/Conv1D/Squeeze:output:02conv1d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` ª
2conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv1d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Î
#conv1d_4/squeeze_batch_dims/BiasAddBiasAdd,conv1d_4/squeeze_batch_dims/Reshape:output:0:conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` |
+conv1d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"`       r
'conv1d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv1d_4/squeeze_batch_dims/concatConcatV22conv1d_4/squeeze_batch_dims/strided_slice:output:04conv1d_4/squeeze_batch_dims/concat/values_1:output:00conv1d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Å
%conv1d_4/squeeze_batch_dims/Reshape_1Reshape,conv1d_4/squeeze_batch_dims/BiasAdd:output:0+conv1d_4/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
re_lu_2/Relu_1Relu.conv1d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` i
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ­
conv1d_5/Conv1D/ExpandDims
ExpandDimsre_lu_2/Relu_1:activations:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` ¤
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  h
conv1d_5/Conv1D/ShapeShape#conv1d_5/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:m
#conv1d_5/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv1d_5/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv1d_5/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv1d_5/Conv1D/strided_sliceStridedSliceconv1d_5/Conv1D/Shape:output:0,conv1d_5/Conv1D/strided_slice/stack:output:0.conv1d_5/Conv1D/strided_slice/stack_1:output:0.conv1d_5/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv1d_5/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   `       ©
conv1d_5/Conv1D/ReshapeReshape#conv1d_5/Conv1D/ExpandDims:output:0&conv1d_5/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` Ì
conv1d_5/Conv1D/Conv2DConv2D conv1d_5/Conv1D/Reshape:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ *
paddingVALID*
strides
t
conv1d_5/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   ^       f
conv1d_5/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv1d_5/Conv1D/concatConcatV2&conv1d_5/Conv1D/strided_slice:output:0(conv1d_5/Conv1D/concat/values_1:output:0$conv1d_5/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv1d_5/Conv1D/Reshape_1Reshapeconv1d_5/Conv1D/Conv2D:output:0conv1d_5/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
^  
conv1d_5/Conv1D/SqueezeSqueeze"conv1d_5/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *
squeeze_dims

ýÿÿÿÿÿÿÿÿq
!conv1d_5/squeeze_batch_dims/ShapeShape conv1d_5/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:y
/conv1d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv1d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ{
1conv1d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv1d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv1d_5/squeeze_batch_dims/Shape:output:08conv1d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv1d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv1d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask~
)conv1d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ^       º
#conv1d_5/squeeze_batch_dims/ReshapeReshape conv1d_5/Conv1D/Squeeze:output:02conv1d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ ª
2conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv1d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Î
#conv1d_5/squeeze_batch_dims/BiasAddBiasAdd,conv1d_5/squeeze_batch_dims/Reshape:output:0:conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ |
+conv1d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"^       r
'conv1d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv1d_5/squeeze_batch_dims/concatConcatV22conv1d_5/squeeze_batch_dims/strided_slice:output:04conv1d_5/squeeze_batch_dims/concat/values_1:output:00conv1d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Å
%conv1d_5/squeeze_batch_dims/Reshape_1Reshape,conv1d_5/squeeze_batch_dims/BiasAdd:output:0+conv1d_5/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ 
re_lu_2/Relu_2Relu.conv1d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ s
IdentityIdentityre_lu_2/Relu_2:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ ï
NoOpNoOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp3^conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp3^conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp3^conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ
d: : : : : : 2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2h
2conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2h
2conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2h
2conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d

_user_specified_namex

Ð
H__inference_enc_forward_layer_call_and_return_conditional_losses_1631719
xJ
4conv1d_3_conv1d_expanddims_1_readvariableop_resource: I
;conv1d_3_squeeze_batch_dims_biasadd_readvariableop_resource: J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:  I
;conv1d_4_squeeze_batch_dims_biasadd_readvariableop_resource: J
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:  I
;conv1d_5_squeeze_batch_dims_biasadd_readvariableop_resource: 
identity¢+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp¢2conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp¢+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢2conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢2conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpi
conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d_3/Conv1D/ExpandDims
ExpandDimsx'conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
d¤
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0b
 conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_3/Conv1D/ExpandDims_1
ExpandDims3conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: h
conv1d_3/Conv1D/ShapeShape#conv1d_3/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:m
#conv1d_3/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv1d_3/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv1d_3/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv1d_3/Conv1D/strided_sliceStridedSliceconv1d_3/Conv1D/Shape:output:0,conv1d_3/Conv1D/strided_slice/stack:output:0.conv1d_3/Conv1D/strided_slice/stack_1:output:0.conv1d_3/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv1d_3/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   d      ©
conv1d_3/Conv1D/ReshapeReshape#conv1d_3/Conv1D/ExpandDims:output:0&conv1d_3/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÌ
conv1d_3/Conv1D/Conv2DConv2D conv1d_3/Conv1D/Reshape:output:0%conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb *
paddingVALID*
strides
t
conv1d_3/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   b       f
conv1d_3/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv1d_3/Conv1D/concatConcatV2&conv1d_3/Conv1D/strided_slice:output:0(conv1d_3/Conv1D/concat/values_1:output:0$conv1d_3/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv1d_3/Conv1D/Reshape_1Reshapeconv1d_3/Conv1D/Conv2D:output:0conv1d_3/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b  
conv1d_3/Conv1D/SqueezeSqueeze"conv1d_3/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *
squeeze_dims

ýÿÿÿÿÿÿÿÿq
!conv1d_3/squeeze_batch_dims/ShapeShape conv1d_3/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:y
/conv1d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv1d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ{
1conv1d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv1d_3/squeeze_batch_dims/strided_sliceStridedSlice*conv1d_3/squeeze_batch_dims/Shape:output:08conv1d_3/squeeze_batch_dims/strided_slice/stack:output:0:conv1d_3/squeeze_batch_dims/strided_slice/stack_1:output:0:conv1d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask~
)conv1d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿb       º
#conv1d_3/squeeze_batch_dims/ReshapeReshape conv1d_3/Conv1D/Squeeze:output:02conv1d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb ª
2conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv1d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Î
#conv1d_3/squeeze_batch_dims/BiasAddBiasAdd,conv1d_3/squeeze_batch_dims/Reshape:output:0:conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb |
+conv1d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"b       r
'conv1d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv1d_3/squeeze_batch_dims/concatConcatV22conv1d_3/squeeze_batch_dims/strided_slice:output:04conv1d_3/squeeze_batch_dims/concat/values_1:output:00conv1d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Å
%conv1d_3/squeeze_batch_dims/Reshape_1Reshape,conv1d_3/squeeze_batch_dims/BiasAdd:output:0+conv1d_3/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b ~
re_lu_2/ReluRelu.conv1d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b i
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ«
conv1d_4/Conv1D/ExpandDims
ExpandDimsre_lu_2/Relu:activations:0'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b ¤
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  h
conv1d_4/Conv1D/ShapeShape#conv1d_4/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:m
#conv1d_4/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv1d_4/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv1d_4/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv1d_4/Conv1D/strided_sliceStridedSliceconv1d_4/Conv1D/Shape:output:0,conv1d_4/Conv1D/strided_slice/stack:output:0.conv1d_4/Conv1D/strided_slice/stack_1:output:0.conv1d_4/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv1d_4/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   b       ©
conv1d_4/Conv1D/ReshapeReshape#conv1d_4/Conv1D/ExpandDims:output:0&conv1d_4/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb Ì
conv1d_4/Conv1D/Conv2DConv2D conv1d_4/Conv1D/Reshape:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` *
paddingVALID*
strides
t
conv1d_4/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   `       f
conv1d_4/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv1d_4/Conv1D/concatConcatV2&conv1d_4/Conv1D/strided_slice:output:0(conv1d_4/Conv1D/concat/values_1:output:0$conv1d_4/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv1d_4/Conv1D/Reshape_1Reshapeconv1d_4/Conv1D/Conv2D:output:0conv1d_4/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
`  
conv1d_4/Conv1D/SqueezeSqueeze"conv1d_4/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *
squeeze_dims

ýÿÿÿÿÿÿÿÿq
!conv1d_4/squeeze_batch_dims/ShapeShape conv1d_4/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:y
/conv1d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv1d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ{
1conv1d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv1d_4/squeeze_batch_dims/strided_sliceStridedSlice*conv1d_4/squeeze_batch_dims/Shape:output:08conv1d_4/squeeze_batch_dims/strided_slice/stack:output:0:conv1d_4/squeeze_batch_dims/strided_slice/stack_1:output:0:conv1d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask~
)conv1d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ`       º
#conv1d_4/squeeze_batch_dims/ReshapeReshape conv1d_4/Conv1D/Squeeze:output:02conv1d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` ª
2conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv1d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Î
#conv1d_4/squeeze_batch_dims/BiasAddBiasAdd,conv1d_4/squeeze_batch_dims/Reshape:output:0:conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` |
+conv1d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"`       r
'conv1d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv1d_4/squeeze_batch_dims/concatConcatV22conv1d_4/squeeze_batch_dims/strided_slice:output:04conv1d_4/squeeze_batch_dims/concat/values_1:output:00conv1d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Å
%conv1d_4/squeeze_batch_dims/Reshape_1Reshape,conv1d_4/squeeze_batch_dims/BiasAdd:output:0+conv1d_4/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
re_lu_2/Relu_1Relu.conv1d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` i
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ­
conv1d_5/Conv1D/ExpandDims
ExpandDimsre_lu_2/Relu_1:activations:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` ¤
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  h
conv1d_5/Conv1D/ShapeShape#conv1d_5/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:m
#conv1d_5/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%conv1d_5/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿo
%conv1d_5/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
conv1d_5/Conv1D/strided_sliceStridedSliceconv1d_5/Conv1D/Shape:output:0,conv1d_5/Conv1D/strided_slice/stack:output:0.conv1d_5/Conv1D/strided_slice/stack_1:output:0.conv1d_5/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskv
conv1d_5/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   `       ©
conv1d_5/Conv1D/ReshapeReshape#conv1d_5/Conv1D/ExpandDims:output:0&conv1d_5/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` Ì
conv1d_5/Conv1D/Conv2DConv2D conv1d_5/Conv1D/Reshape:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ *
paddingVALID*
strides
t
conv1d_5/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   ^       f
conv1d_5/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÈ
conv1d_5/Conv1D/concatConcatV2&conv1d_5/Conv1D/strided_slice:output:0(conv1d_5/Conv1D/concat/values_1:output:0$conv1d_5/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
conv1d_5/Conv1D/Reshape_1Reshapeconv1d_5/Conv1D/Conv2D:output:0conv1d_5/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
^  
conv1d_5/Conv1D/SqueezeSqueeze"conv1d_5/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *
squeeze_dims

ýÿÿÿÿÿÿÿÿq
!conv1d_5/squeeze_batch_dims/ShapeShape conv1d_5/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:y
/conv1d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1conv1d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ{
1conv1d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)conv1d_5/squeeze_batch_dims/strided_sliceStridedSlice*conv1d_5/squeeze_batch_dims/Shape:output:08conv1d_5/squeeze_batch_dims/strided_slice/stack:output:0:conv1d_5/squeeze_batch_dims/strided_slice/stack_1:output:0:conv1d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask~
)conv1d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ^       º
#conv1d_5/squeeze_batch_dims/ReshapeReshape conv1d_5/Conv1D/Squeeze:output:02conv1d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ ª
2conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv1d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Î
#conv1d_5/squeeze_batch_dims/BiasAddBiasAdd,conv1d_5/squeeze_batch_dims/Reshape:output:0:conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ |
+conv1d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"^       r
'conv1d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"conv1d_5/squeeze_batch_dims/concatConcatV22conv1d_5/squeeze_batch_dims/strided_slice:output:04conv1d_5/squeeze_batch_dims/concat/values_1:output:00conv1d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:Å
%conv1d_5/squeeze_batch_dims/Reshape_1Reshape,conv1d_5/squeeze_batch_dims/BiasAdd:output:0+conv1d_5/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ 
re_lu_2/Relu_2Relu.conv1d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ s
IdentityIdentityre_lu_2/Relu_2:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ ï
NoOpNoOp,^conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp3^conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp3^conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp3^conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ
d: : : : : : 2Z
+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2h
2conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2h
2conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2h
2conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d

_user_specified_namex
Ë
û
D__inference_dense_8_layer_call_and_return_conditional_losses_1631961

inputs3
!tensordot_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@
 
_user_specified_nameinputs

û
H__inference_spatial_att_layer_call_and_return_conditional_losses_1631445	
query
feature9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:;
)dense_3_tensordot_readvariableop_resource:;
)dense_2_tensordot_readvariableop_resource: ;
)dense_4_tensordot_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp¢ dense_3/Tensordot/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          L
dense/Tensordot/ShapeShapefeature*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposefeaturedense/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
re_lu_1/ReluReludense/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          a
dense_1/Tensordot/ShapeShapere_lu_1/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¡
dense_1/Tensordot/transpose	Transposere_lu_1/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdj
re_lu_1/Relu_1Reludense_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          c
dense_3/Tensordot/ShapeShapere_lu_1/Relu_1:activations:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:£
dense_3/Tensordot/transpose	Transposere_lu_1/Relu_1:activations:0!dense_3/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       L
dense_2/Tensordot/ShapeShapequery*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/transpose	Transposequery!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ¢
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsdense_2/Tensordot:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
addAddV2dense_3/Tensordot:output:0ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dZ
activation/TanhTanhadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          Z
dense_4/Tensordot/ShapeShapeactivation/Tanh:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposeactivation/Tanh:y:0!dense_4/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d¢
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
SqueezeSqueezedense_4/Tensordot:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*
squeeze_dims

ÿÿÿÿÿÿÿÿÿb
softmax/SoftmaxSoftmaxSqueeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dl
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d³
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
 :ÿÿÿÿÿÿÿÿÿd: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp:R N
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 

_user_specified_namequery:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	feature
ñ

E__inference_conv1d_1_layer_call_and_return_conditional_losses_1628693

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ø
`
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629352

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
ï

C__inference_conv1d_layer_call_and_return_conditional_losses_1628665

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

*__inference_enc_past_layer_call_fn_1628867
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_enc_past_layer_call_and_return_conditional_losses_1628835s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
È
`
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629304

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¤
©
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629223
input_1&
conv1d_3_1629204: 
conv1d_3_1629206: &
conv1d_4_1629210:  
conv1d_4_1629212: &
conv1d_5_1629216:  
conv1d_5_1629218: 
identity¢ conv1d_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCall
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_3_1629204conv1d_3_1629206*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1628955ë
re_lu_2/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1628966
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv1d_4_1629210conv1d_4_1629212*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1629005í
re_lu_2/PartitionedCall_1PartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629015 
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall"re_lu_2/PartitionedCall_1:output:0conv1d_5_1629216conv1d_5_1629218*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1629054í
re_lu_2/PartitionedCall_2PartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629064y
IdentityIdentity"re_lu_2/PartitionedCall_2:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ ¯
NoOpNoOp!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ
d: : : : : : 2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
!
_user_specified_name	input_1
Ý

(__inference_conv1d_layer_call_fn_1632110

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_1628665s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629015

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
 
_user_specified_nameinputs
¾
`
D__inference_flatten_layer_call_and_return_conditional_losses_1629281

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ö
^
B__inference_re_lu_layer_call_and_return_conditional_losses_1632061

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
á

*__inference_conv1d_2_layer_call_fn_1632158

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1628720s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ñ

*__inference_conv1d_5_layer_call_fn_1632274

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1629054w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
` : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
 
_user_specified_nameinputs
°
E
)__inference_flatten_layer_call_fn_1632035

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1629281a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ñ

*__inference_conv1d_3_layer_call_fn_1632182

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1628955w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
d: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
 
_user_specified_nameinputs
è
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1632101

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
^ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ 
 
_user_specified_nameinputs
ñ

E__inference_conv1d_2_layer_call_and_return_conditional_losses_1632173

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý)
º
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1629054

inputsA
+conv1d_expanddims_1_readvariableop_resource:  @
2squeeze_batch_dims_biasadd_readvariableop_resource: 
identity¢"Conv1D/ExpandDims_1/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  V
Conv1D/ShapeShapeConv1D/ExpandDims:output:0*
T0*
_output_shapes
:d
Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿf
Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Conv1D/strided_sliceStridedSliceConv1D/Shape:output:0#Conv1D/strided_slice/stack:output:0%Conv1D/strided_slice/stack_1:output:0%Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   `       
Conv1D/ReshapeReshapeConv1D/ExpandDims:output:0Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` ±
Conv1D/Conv2DConv2DConv1D/Reshape:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ *
paddingVALID*
strides
k
Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   ^       ]
Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
Conv1D/concatConcatV2Conv1D/strided_slice:output:0Conv1D/concat/values_1:output:0Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:
Conv1D/Reshape_1ReshapeConv1D/Conv2D:output:0Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
^ 
Conv1D/SqueezeSqueezeConv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ_
squeeze_batch_dims/ShapeShapeConv1D/Squeeze:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿr
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ^       
squeeze_batch_dims/ReshapeReshapeConv1D/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ 
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ s
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"^       i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÔ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ª
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ |
IdentityIdentity%squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ 
NoOpNoOp#^Conv1D/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
` : : 2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
 
_user_specified_nameinputs
¶
E
)__inference_re_lu_3_layer_call_fn_1632005

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629352d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs

¸
E__inference_conv_att_layer_call_and_return_conditional_losses_1631089

x_past
	x_forward
pos_enc_past
pos_enc_fwdQ
;enc_past_conv1d_conv1d_expanddims_1_readvariableop_resource: =
/enc_past_conv1d_biasadd_readvariableop_resource: S
=enc_past_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  ?
1enc_past_conv1d_1_biasadd_readvariableop_resource: S
=enc_past_conv1d_2_conv1d_expanddims_1_readvariableop_resource:  ?
1enc_past_conv1d_2_biasadd_readvariableop_resource: 9
&dense_5_matmul_readvariableop_resource:	À 5
'dense_5_biasadd_readvariableop_resource: ;
)dense_6_tensordot_readvariableop_resource:$ 5
'dense_6_biasadd_readvariableop_resource: E
3spatial_att_dense_tensordot_readvariableop_resource:?
1spatial_att_dense_biasadd_readvariableop_resource:G
5spatial_att_dense_1_tensordot_readvariableop_resource:A
3spatial_att_dense_1_biasadd_readvariableop_resource:G
5spatial_att_dense_3_tensordot_readvariableop_resource:G
5spatial_att_dense_2_tensordot_readvariableop_resource: G
5spatial_att_dense_4_tensordot_readvariableop_resource:V
@enc_forward_conv1d_3_conv1d_expanddims_1_readvariableop_resource: U
Genc_forward_conv1d_3_squeeze_batch_dims_biasadd_readvariableop_resource: V
@enc_forward_conv1d_4_conv1d_expanddims_1_readvariableop_resource:  U
Genc_forward_conv1d_4_squeeze_batch_dims_biasadd_readvariableop_resource: V
@enc_forward_conv1d_5_conv1d_expanddims_1_readvariableop_resource:  U
Genc_forward_conv1d_5_squeeze_batch_dims_biasadd_readvariableop_resource: <
)dense_7_tensordot_readvariableop_resource:	À 5
'dense_7_biasadd_readvariableop_resource: ;
)dense_8_tensordot_readvariableop_resource:@ 5
'dense_8_biasadd_readvariableop_resource: ;
)dense_9_tensordot_readvariableop_resource: 5
'dense_9_biasadd_readvariableop_resource:
identity¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢ dense_6/Tensordot/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢ dense_7/Tensordot/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢ dense_8/Tensordot/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢ dense_9/Tensordot/ReadVariableOp¢7enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp¢>enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp¢7enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢>enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢7enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢>enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp¢&enc_past/conv1d/BiasAdd/ReadVariableOp¢2enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢(enc_past/conv1d_1/BiasAdd/ReadVariableOp¢4enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢(enc_past/conv1d_2/BiasAdd/ReadVariableOp¢4enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp¢(spatial_att/dense/BiasAdd/ReadVariableOp¢*spatial_att/dense/Tensordot/ReadVariableOp¢*spatial_att/dense_1/BiasAdd/ReadVariableOp¢,spatial_att/dense_1/Tensordot/ReadVariableOp¢,spatial_att/dense_2/Tensordot/ReadVariableOp¢,spatial_att/dense_3/Tensordot/ReadVariableOp¢,spatial_att/dense_4/Tensordot/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :v

ExpandDims
ExpandDims	x_forwardExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :|
ExpandDims_1
ExpandDimspos_enc_fwdExpandDims_1/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
%enc_past/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¡
!enc_past/conv1d/Conv1D/ExpandDims
ExpandDimsx_past.enc_past/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp;enc_past_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0i
'enc_past/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ð
#enc_past/conv1d/Conv1D/ExpandDims_1
ExpandDims:enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:00enc_past/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Ý
enc_past/conv1d/Conv1DConv2D*enc_past/conv1d/Conv1D/ExpandDims:output:0,enc_past/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
 
enc_past/conv1d/Conv1D/SqueezeSqueezeenc_past/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
&enc_past/conv1d/BiasAdd/ReadVariableOpReadVariableOp/enc_past_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0±
enc_past/conv1d/BiasAddBiasAdd'enc_past/conv1d/Conv1D/Squeeze:output:0.enc_past/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
enc_past/re_lu/ReluRelu enc_past/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
'enc_past/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÀ
#enc_past/conv1d_1/Conv1D/ExpandDims
ExpandDims!enc_past/re_lu/Relu:activations:00enc_past/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¶
4enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=enc_past_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0k
)enc_past/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ö
%enc_past/conv1d_1/Conv1D/ExpandDims_1
ExpandDims<enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:02enc_past/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ã
enc_past/conv1d_1/Conv1DConv2D,enc_past/conv1d_1/Conv1D/ExpandDims:output:0.enc_past/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
¤
 enc_past/conv1d_1/Conv1D/SqueezeSqueeze!enc_past/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
(enc_past/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp1enc_past_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0·
enc_past/conv1d_1/BiasAddBiasAdd)enc_past/conv1d_1/Conv1D/Squeeze:output:00enc_past/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
enc_past/re_lu/Relu_1Relu"enc_past/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
'enc_past/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÂ
#enc_past/conv1d_2/Conv1D/ExpandDims
ExpandDims#enc_past/re_lu/Relu_1:activations:00enc_past/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¶
4enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=enc_past_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0k
)enc_past/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ö
%enc_past/conv1d_2/Conv1D/ExpandDims_1
ExpandDims<enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:02enc_past/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ã
enc_past/conv1d_2/Conv1DConv2D,enc_past/conv1d_2/Conv1D/ExpandDims:output:0.enc_past/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
¤
 enc_past/conv1d_2/Conv1D/SqueezeSqueeze!enc_past/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
(enc_past/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp1enc_past_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0·
enc_past/conv1d_2/BiasAddBiasAdd)enc_past/conv1d_2/Conv1D/Squeeze:output:00enc_past/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
enc_past/re_lu/Relu_2Relu"enc_past/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  
flatten/ReshapeReshape#enc_past/re_lu/Relu_2:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	À *
dtype0
dense_5/MatMulMatMulflatten/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
re_lu_3/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsre_lu_3/Relu:activations:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Tile/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"   
      r
TileTileExpandDims_2:output:0Tile/multiples:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
concatConcatV2Tile:output:0pos_enc_pastconcat/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes

:$ *
dtype0`
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       V
dense_6/Tensordot/ShapeShapeconcat:output:0*
T0*
_output_shapes
:a
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_6/Tensordot/transpose	Transposeconcat:output:0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$¢
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 f
re_lu_3/Relu_1Reludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 X
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
concat_1ConcatV2ExpandDims:output:0ExpandDims_1:output:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
*spatial_att/dense/Tensordot/ReadVariableOpReadVariableOp3spatial_att_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0j
 spatial_att/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
 spatial_att/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          b
!spatial_att/dense/Tensordot/ShapeShapeconcat_1:output:0*
T0*
_output_shapes
:k
)spatial_att/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
$spatial_att/dense/Tensordot/GatherV2GatherV2*spatial_att/dense/Tensordot/Shape:output:0)spatial_att/dense/Tensordot/free:output:02spatial_att/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+spatial_att/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&spatial_att/dense/Tensordot/GatherV2_1GatherV2*spatial_att/dense/Tensordot/Shape:output:0)spatial_att/dense/Tensordot/axes:output:04spatial_att/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!spatial_att/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¤
 spatial_att/dense/Tensordot/ProdProd-spatial_att/dense/Tensordot/GatherV2:output:0*spatial_att/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#spatial_att/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ª
"spatial_att/dense/Tensordot/Prod_1Prod/spatial_att/dense/Tensordot/GatherV2_1:output:0,spatial_att/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'spatial_att/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ä
"spatial_att/dense/Tensordot/concatConcatV2)spatial_att/dense/Tensordot/free:output:0)spatial_att/dense/Tensordot/axes:output:00spatial_att/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¯
!spatial_att/dense/Tensordot/stackPack)spatial_att/dense/Tensordot/Prod:output:0+spatial_att/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
%spatial_att/dense/Tensordot/transpose	Transposeconcat_1:output:0+spatial_att/dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÀ
#spatial_att/dense/Tensordot/ReshapeReshape)spatial_att/dense/Tensordot/transpose:y:0*spatial_att/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
"spatial_att/dense/Tensordot/MatMulMatMul,spatial_att/dense/Tensordot/Reshape:output:02spatial_att/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
#spatial_att/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:k
)spatial_att/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ï
$spatial_att/dense/Tensordot/concat_1ConcatV2-spatial_att/dense/Tensordot/GatherV2:output:0,spatial_att/dense/Tensordot/Const_2:output:02spatial_att/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
spatial_att/dense/TensordotReshape,spatial_att/dense/Tensordot/MatMul:product:0-spatial_att/dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(spatial_att/dense/BiasAdd/ReadVariableOpReadVariableOp1spatial_att_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
spatial_att/dense/BiasAddBiasAdd$spatial_att/dense/Tensordot:output:00spatial_att/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
spatial_att/re_lu_1/ReluRelu"spatial_att/dense/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
,spatial_att/dense_1/Tensordot/ReadVariableOpReadVariableOp5spatial_att_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0l
"spatial_att/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:w
"spatial_att/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          y
#spatial_att/dense_1/Tensordot/ShapeShape&spatial_att/re_lu_1/Relu:activations:0*
T0*
_output_shapes
:m
+spatial_att/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&spatial_att/dense_1/Tensordot/GatherV2GatherV2,spatial_att/dense_1/Tensordot/Shape:output:0+spatial_att/dense_1/Tensordot/free:output:04spatial_att/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-spatial_att/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(spatial_att/dense_1/Tensordot/GatherV2_1GatherV2,spatial_att/dense_1/Tensordot/Shape:output:0+spatial_att/dense_1/Tensordot/axes:output:06spatial_att/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#spatial_att/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
"spatial_att/dense_1/Tensordot/ProdProd/spatial_att/dense_1/Tensordot/GatherV2:output:0,spatial_att/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%spatial_att/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: °
$spatial_att/dense_1/Tensordot/Prod_1Prod1spatial_att/dense_1/Tensordot/GatherV2_1:output:0.spatial_att/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)spatial_att/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ì
$spatial_att/dense_1/Tensordot/concatConcatV2+spatial_att/dense_1/Tensordot/free:output:0+spatial_att/dense_1/Tensordot/axes:output:02spatial_att/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:µ
#spatial_att/dense_1/Tensordot/stackPack+spatial_att/dense_1/Tensordot/Prod:output:0-spatial_att/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Å
'spatial_att/dense_1/Tensordot/transpose	Transpose&spatial_att/re_lu_1/Relu:activations:0-spatial_att/dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÆ
%spatial_att/dense_1/Tensordot/ReshapeReshape+spatial_att/dense_1/Tensordot/transpose:y:0,spatial_att/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
$spatial_att/dense_1/Tensordot/MatMulMatMul.spatial_att/dense_1/Tensordot/Reshape:output:04spatial_att/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
%spatial_att/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:m
+spatial_att/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
&spatial_att/dense_1/Tensordot/concat_1ConcatV2/spatial_att/dense_1/Tensordot/GatherV2:output:0.spatial_att/dense_1/Tensordot/Const_2:output:04spatial_att/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ã
spatial_att/dense_1/TensordotReshape.spatial_att/dense_1/Tensordot/MatMul:product:0/spatial_att/dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
*spatial_att/dense_1/BiasAdd/ReadVariableOpReadVariableOp3spatial_att_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
spatial_att/dense_1/BiasAddBiasAdd&spatial_att/dense_1/Tensordot:output:02spatial_att/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
spatial_att/re_lu_1/Relu_1Relu$spatial_att/dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
,spatial_att/dense_3/Tensordot/ReadVariableOpReadVariableOp5spatial_att_dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0l
"spatial_att/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:w
"spatial_att/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          {
#spatial_att/dense_3/Tensordot/ShapeShape(spatial_att/re_lu_1/Relu_1:activations:0*
T0*
_output_shapes
:m
+spatial_att/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&spatial_att/dense_3/Tensordot/GatherV2GatherV2,spatial_att/dense_3/Tensordot/Shape:output:0+spatial_att/dense_3/Tensordot/free:output:04spatial_att/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-spatial_att/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(spatial_att/dense_3/Tensordot/GatherV2_1GatherV2,spatial_att/dense_3/Tensordot/Shape:output:0+spatial_att/dense_3/Tensordot/axes:output:06spatial_att/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#spatial_att/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
"spatial_att/dense_3/Tensordot/ProdProd/spatial_att/dense_3/Tensordot/GatherV2:output:0,spatial_att/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%spatial_att/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: °
$spatial_att/dense_3/Tensordot/Prod_1Prod1spatial_att/dense_3/Tensordot/GatherV2_1:output:0.spatial_att/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)spatial_att/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ì
$spatial_att/dense_3/Tensordot/concatConcatV2+spatial_att/dense_3/Tensordot/free:output:0+spatial_att/dense_3/Tensordot/axes:output:02spatial_att/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:µ
#spatial_att/dense_3/Tensordot/stackPack+spatial_att/dense_3/Tensordot/Prod:output:0-spatial_att/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ç
'spatial_att/dense_3/Tensordot/transpose	Transpose(spatial_att/re_lu_1/Relu_1:activations:0-spatial_att/dense_3/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÆ
%spatial_att/dense_3/Tensordot/ReshapeReshape+spatial_att/dense_3/Tensordot/transpose:y:0,spatial_att/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
$spatial_att/dense_3/Tensordot/MatMulMatMul.spatial_att/dense_3/Tensordot/Reshape:output:04spatial_att/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
%spatial_att/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:m
+spatial_att/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
&spatial_att/dense_3/Tensordot/concat_1ConcatV2/spatial_att/dense_3/Tensordot/GatherV2:output:0.spatial_att/dense_3/Tensordot/Const_2:output:04spatial_att/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ã
spatial_att/dense_3/TensordotReshape.spatial_att/dense_3/Tensordot/MatMul:product:0/spatial_att/dense_3/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
,spatial_att/dense_2/Tensordot/ReadVariableOpReadVariableOp5spatial_att_dense_2_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0l
"spatial_att/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"spatial_att/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       o
#spatial_att/dense_2/Tensordot/ShapeShapere_lu_3/Relu_1:activations:0*
T0*
_output_shapes
:m
+spatial_att/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&spatial_att/dense_2/Tensordot/GatherV2GatherV2,spatial_att/dense_2/Tensordot/Shape:output:0+spatial_att/dense_2/Tensordot/free:output:04spatial_att/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-spatial_att/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(spatial_att/dense_2/Tensordot/GatherV2_1GatherV2,spatial_att/dense_2/Tensordot/Shape:output:0+spatial_att/dense_2/Tensordot/axes:output:06spatial_att/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#spatial_att/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
"spatial_att/dense_2/Tensordot/ProdProd/spatial_att/dense_2/Tensordot/GatherV2:output:0,spatial_att/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%spatial_att/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: °
$spatial_att/dense_2/Tensordot/Prod_1Prod1spatial_att/dense_2/Tensordot/GatherV2_1:output:0.spatial_att/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)spatial_att/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ì
$spatial_att/dense_2/Tensordot/concatConcatV2+spatial_att/dense_2/Tensordot/free:output:0+spatial_att/dense_2/Tensordot/axes:output:02spatial_att/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:µ
#spatial_att/dense_2/Tensordot/stackPack+spatial_att/dense_2/Tensordot/Prod:output:0-spatial_att/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:·
'spatial_att/dense_2/Tensordot/transpose	Transposere_lu_3/Relu_1:activations:0-spatial_att/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Æ
%spatial_att/dense_2/Tensordot/ReshapeReshape+spatial_att/dense_2/Tensordot/transpose:y:0,spatial_att/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
$spatial_att/dense_2/Tensordot/MatMulMatMul.spatial_att/dense_2/Tensordot/Reshape:output:04spatial_att/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
%spatial_att/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:m
+spatial_att/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
&spatial_att/dense_2/Tensordot/concat_1ConcatV2/spatial_att/dense_2/Tensordot/GatherV2:output:0.spatial_att/dense_2/Tensordot/Const_2:output:04spatial_att/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¿
spatial_att/dense_2/TensordotReshape.spatial_att/dense_2/Tensordot/MatMul:product:0/spatial_att/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
\
spatial_att/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :«
spatial_att/ExpandDims
ExpandDims&spatial_att/dense_2/Tensordot:output:0#spatial_att/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

spatial_att/addAddV2&spatial_att/dense_3/Tensordot:output:0spatial_att/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dr
spatial_att/activation/TanhTanhspatial_att/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d¢
,spatial_att/dense_4/Tensordot/ReadVariableOpReadVariableOp5spatial_att_dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0l
"spatial_att/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:w
"spatial_att/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          r
#spatial_att/dense_4/Tensordot/ShapeShapespatial_att/activation/Tanh:y:0*
T0*
_output_shapes
:m
+spatial_att/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&spatial_att/dense_4/Tensordot/GatherV2GatherV2,spatial_att/dense_4/Tensordot/Shape:output:0+spatial_att/dense_4/Tensordot/free:output:04spatial_att/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-spatial_att/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(spatial_att/dense_4/Tensordot/GatherV2_1GatherV2,spatial_att/dense_4/Tensordot/Shape:output:0+spatial_att/dense_4/Tensordot/axes:output:06spatial_att/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#spatial_att/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
"spatial_att/dense_4/Tensordot/ProdProd/spatial_att/dense_4/Tensordot/GatherV2:output:0,spatial_att/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%spatial_att/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: °
$spatial_att/dense_4/Tensordot/Prod_1Prod1spatial_att/dense_4/Tensordot/GatherV2_1:output:0.spatial_att/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)spatial_att/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ì
$spatial_att/dense_4/Tensordot/concatConcatV2+spatial_att/dense_4/Tensordot/free:output:0+spatial_att/dense_4/Tensordot/axes:output:02spatial_att/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:µ
#spatial_att/dense_4/Tensordot/stackPack+spatial_att/dense_4/Tensordot/Prod:output:0-spatial_att/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¾
'spatial_att/dense_4/Tensordot/transpose	Transposespatial_att/activation/Tanh:y:0-spatial_att/dense_4/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dÆ
%spatial_att/dense_4/Tensordot/ReshapeReshape+spatial_att/dense_4/Tensordot/transpose:y:0,spatial_att/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
$spatial_att/dense_4/Tensordot/MatMulMatMul.spatial_att/dense_4/Tensordot/Reshape:output:04spatial_att/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
%spatial_att/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:m
+spatial_att/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
&spatial_att/dense_4/Tensordot/concat_1ConcatV2/spatial_att/dense_4/Tensordot/GatherV2:output:0.spatial_att/dense_4/Tensordot/Const_2:output:04spatial_att/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ã
spatial_att/dense_4/TensordotReshape.spatial_att/dense_4/Tensordot/MatMul:product:0/spatial_att/dense_4/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
spatial_att/SqueezeSqueeze&spatial_att/dense_4/Tensordot:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*
squeeze_dims

ÿÿÿÿÿÿÿÿÿz
spatial_att/softmax/SoftmaxSoftmaxspatial_att/Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dR
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_3
ExpandDims%spatial_att/softmax/Softmax:softmax:0ExpandDims_3/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dp
mulMulExpandDims_3:output:0ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
du
*enc_forward/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ°
&enc_forward/conv1d_3/Conv1D/ExpandDims
ExpandDimsmul:z:03enc_forward/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
d¼
7enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp@enc_forward_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0n
,enc_forward/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ß
(enc_forward/conv1d_3/Conv1D/ExpandDims_1
ExpandDims?enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:05enc_forward/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
!enc_forward/conv1d_3/Conv1D/ShapeShape/enc_forward/conv1d_3/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:y
/enc_forward/conv1d_3/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1enc_forward/conv1d_3/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1enc_forward/conv1d_3/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)enc_forward/conv1d_3/Conv1D/strided_sliceStridedSlice*enc_forward/conv1d_3/Conv1D/Shape:output:08enc_forward/conv1d_3/Conv1D/strided_slice/stack:output:0:enc_forward/conv1d_3/Conv1D/strided_slice/stack_1:output:0:enc_forward/conv1d_3/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)enc_forward/conv1d_3/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   d      Í
#enc_forward/conv1d_3/Conv1D/ReshapeReshape/enc_forward/conv1d_3/Conv1D/ExpandDims:output:02enc_forward/conv1d_3/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdð
"enc_forward/conv1d_3/Conv1D/Conv2DConv2D,enc_forward/conv1d_3/Conv1D/Reshape:output:01enc_forward/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb *
paddingVALID*
strides

+enc_forward/conv1d_3/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   b       r
'enc_forward/conv1d_3/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"enc_forward/conv1d_3/Conv1D/concatConcatV22enc_forward/conv1d_3/Conv1D/strided_slice:output:04enc_forward/conv1d_3/Conv1D/concat/values_1:output:00enc_forward/conv1d_3/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:È
%enc_forward/conv1d_3/Conv1D/Reshape_1Reshape+enc_forward/conv1d_3/Conv1D/Conv2D:output:0+enc_forward/conv1d_3/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b ¸
#enc_forward/conv1d_3/Conv1D/SqueezeSqueeze.enc_forward/conv1d_3/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
-enc_forward/conv1d_3/squeeze_batch_dims/ShapeShape,enc_forward/conv1d_3/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:
;enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
=enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5enc_forward/conv1d_3/squeeze_batch_dims/strided_sliceStridedSlice6enc_forward/conv1d_3/squeeze_batch_dims/Shape:output:0Denc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack:output:0Fenc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_1:output:0Fenc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
5enc_forward/conv1d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿb       Þ
/enc_forward/conv1d_3/squeeze_batch_dims/ReshapeReshape,enc_forward/conv1d_3/Conv1D/Squeeze:output:0>enc_forward/conv1d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb Â
>enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpGenc_forward_conv1d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ò
/enc_forward/conv1d_3/squeeze_batch_dims/BiasAddBiasAdd8enc_forward/conv1d_3/squeeze_batch_dims/Reshape:output:0Fenc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb 
7enc_forward/conv1d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"b       ~
3enc_forward/conv1d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¨
.enc_forward/conv1d_3/squeeze_batch_dims/concatConcatV2>enc_forward/conv1d_3/squeeze_batch_dims/strided_slice:output:0@enc_forward/conv1d_3/squeeze_batch_dims/concat/values_1:output:0<enc_forward/conv1d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:é
1enc_forward/conv1d_3/squeeze_batch_dims/Reshape_1Reshape8enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd:output:07enc_forward/conv1d_3/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b 
enc_forward/re_lu_2/ReluRelu:enc_forward/conv1d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b u
*enc_forward/conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÏ
&enc_forward/conv1d_4/Conv1D/ExpandDims
ExpandDims&enc_forward/re_lu_2/Relu:activations:03enc_forward/conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b ¼
7enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp@enc_forward_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0n
,enc_forward/conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ß
(enc_forward/conv1d_4/Conv1D/ExpandDims_1
ExpandDims?enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:05enc_forward/conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  
!enc_forward/conv1d_4/Conv1D/ShapeShape/enc_forward/conv1d_4/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:y
/enc_forward/conv1d_4/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1enc_forward/conv1d_4/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1enc_forward/conv1d_4/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)enc_forward/conv1d_4/Conv1D/strided_sliceStridedSlice*enc_forward/conv1d_4/Conv1D/Shape:output:08enc_forward/conv1d_4/Conv1D/strided_slice/stack:output:0:enc_forward/conv1d_4/Conv1D/strided_slice/stack_1:output:0:enc_forward/conv1d_4/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)enc_forward/conv1d_4/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   b       Í
#enc_forward/conv1d_4/Conv1D/ReshapeReshape/enc_forward/conv1d_4/Conv1D/ExpandDims:output:02enc_forward/conv1d_4/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb ð
"enc_forward/conv1d_4/Conv1D/Conv2DConv2D,enc_forward/conv1d_4/Conv1D/Reshape:output:01enc_forward/conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` *
paddingVALID*
strides

+enc_forward/conv1d_4/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   `       r
'enc_forward/conv1d_4/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"enc_forward/conv1d_4/Conv1D/concatConcatV22enc_forward/conv1d_4/Conv1D/strided_slice:output:04enc_forward/conv1d_4/Conv1D/concat/values_1:output:00enc_forward/conv1d_4/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:È
%enc_forward/conv1d_4/Conv1D/Reshape_1Reshape+enc_forward/conv1d_4/Conv1D/Conv2D:output:0+enc_forward/conv1d_4/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` ¸
#enc_forward/conv1d_4/Conv1D/SqueezeSqueeze.enc_forward/conv1d_4/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
-enc_forward/conv1d_4/squeeze_batch_dims/ShapeShape,enc_forward/conv1d_4/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:
;enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
=enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5enc_forward/conv1d_4/squeeze_batch_dims/strided_sliceStridedSlice6enc_forward/conv1d_4/squeeze_batch_dims/Shape:output:0Denc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack:output:0Fenc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_1:output:0Fenc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
5enc_forward/conv1d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ`       Þ
/enc_forward/conv1d_4/squeeze_batch_dims/ReshapeReshape,enc_forward/conv1d_4/Conv1D/Squeeze:output:0>enc_forward/conv1d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` Â
>enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpGenc_forward_conv1d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ò
/enc_forward/conv1d_4/squeeze_batch_dims/BiasAddBiasAdd8enc_forward/conv1d_4/squeeze_batch_dims/Reshape:output:0Fenc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` 
7enc_forward/conv1d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"`       ~
3enc_forward/conv1d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¨
.enc_forward/conv1d_4/squeeze_batch_dims/concatConcatV2>enc_forward/conv1d_4/squeeze_batch_dims/strided_slice:output:0@enc_forward/conv1d_4/squeeze_batch_dims/concat/values_1:output:0<enc_forward/conv1d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:é
1enc_forward/conv1d_4/squeeze_batch_dims/Reshape_1Reshape8enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd:output:07enc_forward/conv1d_4/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
enc_forward/re_lu_2/Relu_1Relu:enc_forward/conv1d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` u
*enc_forward/conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÑ
&enc_forward/conv1d_5/Conv1D/ExpandDims
ExpandDims(enc_forward/re_lu_2/Relu_1:activations:03enc_forward/conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` ¼
7enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp@enc_forward_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0n
,enc_forward/conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ß
(enc_forward/conv1d_5/Conv1D/ExpandDims_1
ExpandDims?enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:05enc_forward/conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  
!enc_forward/conv1d_5/Conv1D/ShapeShape/enc_forward/conv1d_5/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:y
/enc_forward/conv1d_5/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1enc_forward/conv1d_5/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1enc_forward/conv1d_5/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)enc_forward/conv1d_5/Conv1D/strided_sliceStridedSlice*enc_forward/conv1d_5/Conv1D/Shape:output:08enc_forward/conv1d_5/Conv1D/strided_slice/stack:output:0:enc_forward/conv1d_5/Conv1D/strided_slice/stack_1:output:0:enc_forward/conv1d_5/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)enc_forward/conv1d_5/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   `       Í
#enc_forward/conv1d_5/Conv1D/ReshapeReshape/enc_forward/conv1d_5/Conv1D/ExpandDims:output:02enc_forward/conv1d_5/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` ð
"enc_forward/conv1d_5/Conv1D/Conv2DConv2D,enc_forward/conv1d_5/Conv1D/Reshape:output:01enc_forward/conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ *
paddingVALID*
strides

+enc_forward/conv1d_5/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   ^       r
'enc_forward/conv1d_5/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"enc_forward/conv1d_5/Conv1D/concatConcatV22enc_forward/conv1d_5/Conv1D/strided_slice:output:04enc_forward/conv1d_5/Conv1D/concat/values_1:output:00enc_forward/conv1d_5/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:È
%enc_forward/conv1d_5/Conv1D/Reshape_1Reshape+enc_forward/conv1d_5/Conv1D/Conv2D:output:0+enc_forward/conv1d_5/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
^ ¸
#enc_forward/conv1d_5/Conv1D/SqueezeSqueeze.enc_forward/conv1d_5/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
-enc_forward/conv1d_5/squeeze_batch_dims/ShapeShape,enc_forward/conv1d_5/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:
;enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
=enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5enc_forward/conv1d_5/squeeze_batch_dims/strided_sliceStridedSlice6enc_forward/conv1d_5/squeeze_batch_dims/Shape:output:0Denc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack:output:0Fenc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_1:output:0Fenc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
5enc_forward/conv1d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ^       Þ
/enc_forward/conv1d_5/squeeze_batch_dims/ReshapeReshape,enc_forward/conv1d_5/Conv1D/Squeeze:output:0>enc_forward/conv1d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ Â
>enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpGenc_forward_conv1d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ò
/enc_forward/conv1d_5/squeeze_batch_dims/BiasAddBiasAdd8enc_forward/conv1d_5/squeeze_batch_dims/Reshape:output:0Fenc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ 
7enc_forward/conv1d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"^       ~
3enc_forward/conv1d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¨
.enc_forward/conv1d_5/squeeze_batch_dims/concatConcatV2>enc_forward/conv1d_5/squeeze_batch_dims/strided_slice:output:0@enc_forward/conv1d_5/squeeze_batch_dims/concat/values_1:output:0<enc_forward/conv1d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:é
1enc_forward/conv1d_5/squeeze_batch_dims/Reshape_1Reshape8enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd:output:07enc_forward/conv1d_5/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ 
enc_forward/re_lu_2/Relu_2Relu:enc_forward/conv1d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ
   À  
ReshapeReshape(enc_forward/re_lu_2/Relu_2:activations:0Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes
:	À *
dtype0`
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       W
dense_7/Tensordot/ShapeShapeReshape:output:0*
T0*
_output_shapes
:a
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_7/Tensordot/transpose	TransposeReshape:output:0!dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À¢
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 f
re_lu_3/Relu_2Reludense_7/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 X
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ§
concat_2ConcatV2re_lu_3/Relu_1:activations:0re_lu_3/Relu_2:activations:0concat_2/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0`
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       X
dense_8/Tensordot/ShapeShapeconcat_2:output:0*
T0*
_output_shapes
:a
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_8/Tensordot/transpose	Transposeconcat_2:output:0!dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@¢
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 f
re_lu_3/Relu_3Reludense_8/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_9/Tensordot/ShapeShapere_lu_3/Relu_3:activations:0*
T0*
_output_shapes
:a
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_9/Tensordot/transpose	Transposere_lu_3/Relu_3:activations:0!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ¢
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
activation_1/TanhTanhdense_9/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿÿÿÿÿj
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         è
strided_sliceStridedSlicex_paststrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_maskq
addAddV2activation_1/Tanh:y:0strided_slice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ë

NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp8^enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp?^enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp8^enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp?^enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp8^enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp?^enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp'^enc_past/conv1d/BiasAdd/ReadVariableOp3^enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp)^enc_past/conv1d_1/BiasAdd/ReadVariableOp5^enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp)^enc_past/conv1d_2/BiasAdd/ReadVariableOp5^enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp)^spatial_att/dense/BiasAdd/ReadVariableOp+^spatial_att/dense/Tensordot/ReadVariableOp+^spatial_att/dense_1/BiasAdd/ReadVariableOp-^spatial_att/dense_1/Tensordot/ReadVariableOp-^spatial_att/dense_2/Tensordot/ReadVariableOp-^spatial_att/dense_3/Tensordot/ReadVariableOp-^spatial_att/dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*«
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp2r
7enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp7enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2
>enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp>enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2r
7enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp7enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2
>enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp>enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2r
7enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp7enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2
>enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp>enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2P
&enc_past/conv1d/BiasAdd/ReadVariableOp&enc_past/conv1d/BiasAdd/ReadVariableOp2h
2enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2T
(enc_past/conv1d_1/BiasAdd/ReadVariableOp(enc_past/conv1d_1/BiasAdd/ReadVariableOp2l
4enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp4enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2T
(enc_past/conv1d_2/BiasAdd/ReadVariableOp(enc_past/conv1d_2/BiasAdd/ReadVariableOp2l
4enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp4enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2T
(spatial_att/dense/BiasAdd/ReadVariableOp(spatial_att/dense/BiasAdd/ReadVariableOp2X
*spatial_att/dense/Tensordot/ReadVariableOp*spatial_att/dense/Tensordot/ReadVariableOp2X
*spatial_att/dense_1/BiasAdd/ReadVariableOp*spatial_att/dense_1/BiasAdd/ReadVariableOp2\
,spatial_att/dense_1/Tensordot/ReadVariableOp,spatial_att/dense_1/Tensordot/ReadVariableOp2\
,spatial_att/dense_2/Tensordot/ReadVariableOp,spatial_att/dense_2/Tensordot/ReadVariableOp2\
,spatial_att/dense_3/Tensordot/ReadVariableOp,spatial_att/dense_3/Tensordot/ReadVariableOp2\
,spatial_att/dense_4/Tensordot/ReadVariableOp,spatial_att/dense_4/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namex_past:VR
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#
_user_specified_name	x_forward:YU
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&
_user_specified_namepos_enc_past:XT
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namepos_enc_fwd

û
H__inference_spatial_att_layer_call_and_return_conditional_losses_1631579	
query
feature9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:;
)dense_3_tensordot_readvariableop_resource:;
)dense_2_tensordot_readvariableop_resource: ;
)dense_4_tensordot_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp¢ dense_3/Tensordot/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          L
dense/Tensordot/ShapeShapefeature*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposefeaturedense/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
re_lu_1/ReluReludense/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          a
dense_1/Tensordot/ShapeShapere_lu_1/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¡
dense_1/Tensordot/transpose	Transposere_lu_1/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdj
re_lu_1/Relu_1Reludense_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          c
dense_3/Tensordot/ShapeShapere_lu_1/Relu_1:activations:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:£
dense_3/Tensordot/transpose	Transposere_lu_1/Relu_1:activations:0!dense_3/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       L
dense_2/Tensordot/ShapeShapequery*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/transpose	Transposequery!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ¢
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsdense_2/Tensordot:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
addAddV2dense_3/Tensordot:output:0ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dZ
activation/TanhTanhadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          Z
dense_4/Tensordot/ShapeShapeactivation/Tanh:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposeactivation/Tanh:y:0!dense_4/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d¢
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
SqueezeSqueezedense_4/Tensordot:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*
squeeze_dims

ÿÿÿÿÿÿÿÿÿb
softmax/SoftmaxSoftmaxSqueeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dl
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d³
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
 :ÿÿÿÿÿÿÿÿÿd: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp:R N
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 

_user_specified_namequery:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	feature
Ý)
º
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1628955

inputsA
+conv1d_expanddims_1_readvariableop_resource: @
2squeeze_batch_dims_biasadd_readvariableop_resource: 
identity¢"Conv1D/ExpandDims_1/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
d
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: V
Conv1D/ShapeShapeConv1D/ExpandDims:output:0*
T0*
_output_shapes
:d
Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿf
Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Conv1D/strided_sliceStridedSliceConv1D/Shape:output:0#Conv1D/strided_slice/stack:output:0%Conv1D/strided_slice/stack_1:output:0%Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   d      
Conv1D/ReshapeReshapeConv1D/ExpandDims:output:0Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd±
Conv1D/Conv2DConv2DConv1D/Reshape:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb *
paddingVALID*
strides
k
Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   b       ]
Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
Conv1D/concatConcatV2Conv1D/strided_slice:output:0Conv1D/concat/values_1:output:0Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:
Conv1D/Reshape_1ReshapeConv1D/Conv2D:output:0Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b 
Conv1D/SqueezeSqueezeConv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *
squeeze_dims

ýÿÿÿÿÿÿÿÿ_
squeeze_batch_dims/ShapeShapeConv1D/Squeeze:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿr
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿb       
squeeze_batch_dims/ReshapeReshapeConv1D/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb 
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb s
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"b       i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÔ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ª
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b |
IdentityIdentity%squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b 
NoOpNoOp#^Conv1D/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
d: : 2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
 
_user_specified_nameinputs
Ã

E__inference_enc_past_layer_call_and_return_conditional_losses_1628835
x$
conv1d_1628816: 
conv1d_1628818: &
conv1d_1_1628822:  
conv1d_1_1628824: &
conv1d_2_1628828:  
conv1d_2_1628830: 
identity¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCalló
conv1d/StatefulPartitionedCallStatefulPartitionedCallxconv1d_1628816conv1d_1628818*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_1628665á
re_lu/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628676
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv1d_1_1628822conv1d_1_1628824*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1628693å
re_lu/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628703
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_1:output:0conv1d_2_1628828conv1d_2_1628830*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1628720å
re_lu/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628730s
IdentityIdentity re_lu/PartitionedCall_2:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
È
`
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1632015

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ë
û
D__inference_dense_6_layer_call_and_return_conditional_losses_1629342

inputs3
!tensordot_readvariableop_resource:$ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:$ *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
 
_user_specified_nameinputs
Ð
ü
D__inference_dense_7_layer_call_and_return_conditional_losses_1631922

inputs4
!tensordot_readvariableop_resource:	À -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	À *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
À: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
 
_user_specified_nameinputs

û
H__inference_spatial_att_layer_call_and_return_conditional_losses_1629910	
query
feature9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:;
)dense_3_tensordot_readvariableop_resource:;
)dense_2_tensordot_readvariableop_resource: ;
)dense_4_tensordot_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp¢ dense_3/Tensordot/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          L
dense/Tensordot/ShapeShapefeature*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposefeaturedense/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
re_lu_1/ReluReludense/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          a
dense_1/Tensordot/ShapeShapere_lu_1/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¡
dense_1/Tensordot/transpose	Transposere_lu_1/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdj
re_lu_1/Relu_1Reludense_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          c
dense_3/Tensordot/ShapeShapere_lu_1/Relu_1:activations:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:£
dense_3/Tensordot/transpose	Transposere_lu_1/Relu_1:activations:0!dense_3/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       L
dense_2/Tensordot/ShapeShapequery*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/transpose	Transposequery!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ¢
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsdense_2/Tensordot:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
addAddV2dense_3/Tensordot:output:0ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dZ
activation/TanhTanhadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          Z
dense_4/Tensordot/ShapeShapeactivation/Tanh:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposeactivation/Tanh:y:0!dense_4/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d¢
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
SqueezeSqueezedense_4/Tensordot:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*
squeeze_dims

ÿÿÿÿÿÿÿÿÿb
softmax/SoftmaxSoftmaxSqueeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dl
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d³
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
 :ÿÿÿÿÿÿÿÿÿd: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp:R N
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 

_user_specified_namequery:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	feature
À
J
.__inference_activation_1_layer_call_fn_1632025

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1629641d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ë
û
D__inference_dense_9_layer_call_and_return_conditional_losses_1629630

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
Ö
^
B__inference_re_lu_layer_call_and_return_conditional_losses_1632071

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ï

C__inference_conv1d_layer_call_and_return_conditional_losses_1632125

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
`
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1632020

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
	

*__inference_enc_past_layer_call_fn_1631174
x
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_enc_past_layer_call_and_return_conditional_losses_1628733s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ë
û
D__inference_dense_9_layer_call_and_return_conditional_losses_1632000

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
ª	

-__inference_enc_forward_layer_call_fn_1629201
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629169w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ
d: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
!
_user_specified_name	input_1
²
C
'__inference_re_lu_layer_call_fn_1632046

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628730d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
[
Æ
E__inference_conv_att_layer_call_and_return_conditional_losses_1630066

x_past
	x_forward
pos_enc_past
pos_enc_fwd&
enc_past_1629973: 
enc_past_1629975: &
enc_past_1629977:  
enc_past_1629979: &
enc_past_1629981:  
enc_past_1629983: "
dense_5_1629987:	À 
dense_5_1629989: !
dense_6_1629999:$ 
dense_6_1630001: %
spatial_att_1630007:!
spatial_att_1630009:%
spatial_att_1630011:!
spatial_att_1630013:%
spatial_att_1630015:%
spatial_att_1630017: %
spatial_att_1630019:)
enc_forward_1630025: !
enc_forward_1630027: )
enc_forward_1630029:  !
enc_forward_1630031: )
enc_forward_1630033:  !
enc_forward_1630035: "
dense_7_1630040:	À 
dense_7_1630042: !
dense_8_1630048:@ 
dense_8_1630050: !
dense_9_1630054: 
dense_9_1630056:
identity¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢#enc_forward/StatefulPartitionedCall¢ enc_past/StatefulPartitionedCall¢#spatial_att/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :v

ExpandDims
ExpandDims	x_forwardExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :|
ExpandDims_1
ExpandDimspos_enc_fwdExpandDims_1/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÐ
 enc_past/StatefulPartitionedCallStatefulPartitionedCallx_pastenc_past_1629973enc_past_1629975enc_past_1629977enc_past_1629979enc_past_1629981enc_past_1629983*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_enc_past_layer_call_and_return_conditional_losses_1628835ä
flatten/PartitionedCallPartitionedCall)enc_past/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1629281
dense_5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_5_1629987dense_5_1629989*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1629293â
re_lu_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629304R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDims re_lu_3/PartitionedCall:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Tile/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"   
      r
TileTileExpandDims_2:output:0Tile/multiples:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
concatConcatV2Tile:output:0pos_enc_pastconcat/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
dense_6/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_6_1629999dense_6_1630001*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1629342è
re_lu_3/PartitionedCall_1PartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629352X
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
concat_1ConcatV2ExpandDims:output:0ExpandDims_1:output:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¯
#spatial_att/StatefulPartitionedCallStatefulPartitionedCall"re_lu_3/PartitionedCall_1:output:0concat_1:output:0spatial_att_1630007spatial_att_1630009spatial_att_1630011spatial_att_1630013spatial_att_1630015spatial_att_1630017spatial_att_1630019*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*)
_read_only_resource_inputs
	*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_spatial_att_layer_call_and_return_conditional_losses_1629910R
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_3
ExpandDims,spatial_att/StatefulPartitionedCall:output:0ExpandDims_3/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dp
mulMulExpandDims_3:output:0ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dí
#enc_forward/StatefulPartitionedCallStatefulPartitionedCallmul:z:0enc_forward_1630025enc_forward_1630027enc_forward_1630029enc_forward_1630031enc_forward_1630033enc_forward_1630035*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629169b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ
   À  
ReshapeReshape,enc_forward/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
dense_7/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_7_1630040dense_7_1630042*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1629554è
re_lu_3/PartitionedCall_2PartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629352X
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ³
concat_2ConcatV2"re_lu_3/PartitionedCall_1:output:0"re_lu_3/PartitionedCall_2:output:0concat_2/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@
dense_8/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0dense_8_1630048dense_8_1630050*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1629593è
re_lu_3/PartitionedCall_3PartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629352
dense_9/StatefulPartitionedCallStatefulPartitionedCall"re_lu_3/PartitionedCall_3:output:0dense_9_1630054dense_9_1630056*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1629630ð
activation_1/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1629641h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿÿÿÿÿj
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         è
strided_sliceStridedSlicex_paststrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
addAddV2%activation_1/PartitionedCall:output:0strided_slice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ß
NoOpNoOp ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall$^enc_forward/StatefulPartitionedCall!^enc_past/StatefulPartitionedCall$^spatial_att/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*«
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2J
#enc_forward/StatefulPartitionedCall#enc_forward/StatefulPartitionedCall2D
 enc_past/StatefulPartitionedCall enc_past/StatefulPartitionedCall2J
#spatial_att/StatefulPartitionedCall#spatial_att/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namex_past:VR
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#
_user_specified_name	x_forward:YU
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&
_user_specified_namepos_enc_past:XT
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namepos_enc_fwd
¿
Ô
*__inference_conv_att_layer_call_fn_1630259

x_past
	x_forward
pos_enc_past
pos_enc_fwd
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:	À 
	unknown_6: 
	unknown_7:$ 
	unknown_8: 
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17:  

unknown_18:  

unknown_19:  

unknown_20:  

unknown_21: 

unknown_22:	À 

unknown_23: 

unknown_24:@ 

unknown_25: 

unknown_26: 

unknown_27:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallx_past	x_forwardpos_enc_pastpos_enc_fwdunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*?
_read_only_resource_inputs!
	
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv_att_layer_call_and_return_conditional_losses_1630066s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*«
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namex_past:VR
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#
_user_specified_name	x_forward:YU
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&
_user_specified_namepos_enc_past:XT
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namepos_enc_fwd
ñ

*__inference_conv1d_4_layer_call_fn_1632228

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1629005w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
b : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b 
 
_user_specified_nameinputs

£
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629169
x&
conv1d_3_1629150: 
conv1d_3_1629152: &
conv1d_4_1629156:  
conv1d_4_1629158: &
conv1d_5_1629162:  
conv1d_5_1629164: 
identity¢ conv1d_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCallÿ
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallxconv1d_3_1629150conv1d_3_1629152*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1628955ë
re_lu_2/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1628966
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv1d_4_1629156conv1d_4_1629158*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1629005í
re_lu_2/PartitionedCall_1PartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629015 
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall"re_lu_2/PartitionedCall_1:output:0conv1d_5_1629162conv1d_5_1629164*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1629054í
re_lu_2/PartitionedCall_2PartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629064y
IdentityIdentity"re_lu_2/PartitionedCall_2:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ ¯
NoOpNoOp!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ
d: : : : : : 2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d

_user_specified_namex
ñ

E__inference_conv1d_1_layer_call_and_return_conditional_losses_1632149

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
[
Æ
E__inference_conv_att_layer_call_and_return_conditional_losses_1629649

x_past
	x_forward
pos_enc_past
pos_enc_fwd&
enc_past_1629262: 
enc_past_1629264: &
enc_past_1629266:  
enc_past_1629268: &
enc_past_1629270:  
enc_past_1629272: "
dense_5_1629294:	À 
dense_5_1629296: !
dense_6_1629343:$ 
dense_6_1629345: %
spatial_att_1629491:!
spatial_att_1629493:%
spatial_att_1629495:!
spatial_att_1629497:%
spatial_att_1629499:%
spatial_att_1629501: %
spatial_att_1629503:)
enc_forward_1629509: !
enc_forward_1629511: )
enc_forward_1629513:  !
enc_forward_1629515: )
enc_forward_1629517:  !
enc_forward_1629519: "
dense_7_1629555:	À 
dense_7_1629557: !
dense_8_1629594:@ 
dense_8_1629596: !
dense_9_1629631: 
dense_9_1629633:
identity¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢#enc_forward/StatefulPartitionedCall¢ enc_past/StatefulPartitionedCall¢#spatial_att/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :v

ExpandDims
ExpandDims	x_forwardExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :|
ExpandDims_1
ExpandDimspos_enc_fwdExpandDims_1/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÐ
 enc_past/StatefulPartitionedCallStatefulPartitionedCallx_pastenc_past_1629262enc_past_1629264enc_past_1629266enc_past_1629268enc_past_1629270enc_past_1629272*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_enc_past_layer_call_and_return_conditional_losses_1628733ä
flatten/PartitionedCallPartitionedCall)enc_past/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1629281
dense_5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_5_1629294dense_5_1629296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1629293â
re_lu_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629304R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDims re_lu_3/PartitionedCall:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Tile/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"   
      r
TileTileExpandDims_2:output:0Tile/multiples:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
concatConcatV2Tile:output:0pos_enc_pastconcat/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
dense_6/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_6_1629343dense_6_1629345*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_1629342è
re_lu_3/PartitionedCall_1PartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629352X
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
concat_1ConcatV2ExpandDims:output:0ExpandDims_1:output:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¯
#spatial_att/StatefulPartitionedCallStatefulPartitionedCall"re_lu_3/PartitionedCall_1:output:0concat_1:output:0spatial_att_1629491spatial_att_1629493spatial_att_1629495spatial_att_1629497spatial_att_1629499spatial_att_1629501spatial_att_1629503*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*)
_read_only_resource_inputs
	*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_spatial_att_layer_call_and_return_conditional_losses_1629490R
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_3
ExpandDims,spatial_att/StatefulPartitionedCall:output:0ExpandDims_3/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dp
mulMulExpandDims_3:output:0ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dí
#enc_forward/StatefulPartitionedCallStatefulPartitionedCallmul:z:0enc_forward_1629509enc_forward_1629511enc_forward_1629513enc_forward_1629515enc_forward_1629517enc_forward_1629519*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629067b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ
   À  
ReshapeReshape,enc_forward/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
dense_7/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_7_1629555dense_7_1629557*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_1629554è
re_lu_3/PartitionedCall_2PartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629352X
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ³
concat_2ConcatV2"re_lu_3/PartitionedCall_1:output:0"re_lu_3/PartitionedCall_2:output:0concat_2/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@
dense_8/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0dense_8_1629594dense_8_1629596*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1629593è
re_lu_3/PartitionedCall_3PartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629352
dense_9/StatefulPartitionedCallStatefulPartitionedCall"re_lu_3/PartitionedCall_3:output:0dense_9_1629631dense_9_1629633*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1629630ð
activation_1/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1629641h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿÿÿÿÿj
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         è
strided_sliceStridedSlicex_paststrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
addAddV2%activation_1/PartitionedCall:output:0strided_slice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ß
NoOpNoOp ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall$^enc_forward/StatefulPartitionedCall!^enc_past/StatefulPartitionedCall$^spatial_att/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*«
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2J
#enc_forward/StatefulPartitionedCall#enc_forward/StatefulPartitionedCall2D
 enc_past/StatefulPartitionedCall enc_past/StatefulPartitionedCall2J
#spatial_att/StatefulPartitionedCall#spatial_att/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namex_past:VR
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#
_user_specified_name	x_forward:YU
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&
_user_specified_namepos_enc_past:XT
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namepos_enc_fwd
Ý)
º
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1632265

inputsA
+conv1d_expanddims_1_readvariableop_resource:  @
2squeeze_batch_dims_biasadd_readvariableop_resource: 
identity¢"Conv1D/ExpandDims_1/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  V
Conv1D/ShapeShapeConv1D/ExpandDims:output:0*
T0*
_output_shapes
:d
Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿf
Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Conv1D/strided_sliceStridedSliceConv1D/Shape:output:0#Conv1D/strided_slice/stack:output:0%Conv1D/strided_slice/stack_1:output:0%Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   b       
Conv1D/ReshapeReshapeConv1D/ExpandDims:output:0Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb ±
Conv1D/Conv2DConv2DConv1D/Reshape:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` *
paddingVALID*
strides
k
Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   `       ]
Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
Conv1D/concatConcatV2Conv1D/strided_slice:output:0Conv1D/concat/values_1:output:0Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:
Conv1D/Reshape_1ReshapeConv1D/Conv2D:output:0Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` 
Conv1D/SqueezeSqueezeConv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *
squeeze_dims

ýÿÿÿÿÿÿÿÿ_
squeeze_batch_dims/ShapeShapeConv1D/Squeeze:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿr
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ`       
squeeze_batch_dims/ReshapeReshapeConv1D/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` 
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` s
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"`       i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÔ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ª
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` |
IdentityIdentity%squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
NoOpNoOp#^Conv1D/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
b : : 2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b 
 
_user_specified_nameinputs
Ë
û
D__inference_dense_8_layer_call_and_return_conditional_losses_1629593

inputs3
!tensordot_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@
 
_user_specified_nameinputs
¤
©
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629245
input_1&
conv1d_3_1629226: 
conv1d_3_1629228: &
conv1d_4_1629232:  
conv1d_4_1629234: &
conv1d_5_1629238:  
conv1d_5_1629240: 
identity¢ conv1d_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCall
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_3_1629226conv1d_3_1629228*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1628955ë
re_lu_2/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1628966
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv1d_4_1629232conv1d_4_1629234*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1629005í
re_lu_2/PartitionedCall_1PartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629015 
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall"re_lu_2/PartitionedCall_1:output:0conv1d_5_1629238conv1d_5_1629240*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1629054í
re_lu_2/PartitionedCall_2PartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629064y
IdentityIdentity"re_lu_2/PartitionedCall_2:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ ¯
NoOpNoOp!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ
d: : : : : : 2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
!
_user_specified_name	input_1
	

-__inference_enc_forward_layer_call_fn_1631613
x
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629169w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ
d: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d

_user_specified_namex
ñ

E__inference_conv1d_2_layer_call_and_return_conditional_losses_1628720

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¸
E__inference_conv_att_layer_call_and_return_conditional_losses_1630674

x_past
	x_forward
pos_enc_past
pos_enc_fwdQ
;enc_past_conv1d_conv1d_expanddims_1_readvariableop_resource: =
/enc_past_conv1d_biasadd_readvariableop_resource: S
=enc_past_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  ?
1enc_past_conv1d_1_biasadd_readvariableop_resource: S
=enc_past_conv1d_2_conv1d_expanddims_1_readvariableop_resource:  ?
1enc_past_conv1d_2_biasadd_readvariableop_resource: 9
&dense_5_matmul_readvariableop_resource:	À 5
'dense_5_biasadd_readvariableop_resource: ;
)dense_6_tensordot_readvariableop_resource:$ 5
'dense_6_biasadd_readvariableop_resource: E
3spatial_att_dense_tensordot_readvariableop_resource:?
1spatial_att_dense_biasadd_readvariableop_resource:G
5spatial_att_dense_1_tensordot_readvariableop_resource:A
3spatial_att_dense_1_biasadd_readvariableop_resource:G
5spatial_att_dense_3_tensordot_readvariableop_resource:G
5spatial_att_dense_2_tensordot_readvariableop_resource: G
5spatial_att_dense_4_tensordot_readvariableop_resource:V
@enc_forward_conv1d_3_conv1d_expanddims_1_readvariableop_resource: U
Genc_forward_conv1d_3_squeeze_batch_dims_biasadd_readvariableop_resource: V
@enc_forward_conv1d_4_conv1d_expanddims_1_readvariableop_resource:  U
Genc_forward_conv1d_4_squeeze_batch_dims_biasadd_readvariableop_resource: V
@enc_forward_conv1d_5_conv1d_expanddims_1_readvariableop_resource:  U
Genc_forward_conv1d_5_squeeze_batch_dims_biasadd_readvariableop_resource: <
)dense_7_tensordot_readvariableop_resource:	À 5
'dense_7_biasadd_readvariableop_resource: ;
)dense_8_tensordot_readvariableop_resource:@ 5
'dense_8_biasadd_readvariableop_resource: ;
)dense_9_tensordot_readvariableop_resource: 5
'dense_9_biasadd_readvariableop_resource:
identity¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢ dense_6/Tensordot/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢ dense_7/Tensordot/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢ dense_8/Tensordot/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢ dense_9/Tensordot/ReadVariableOp¢7enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp¢>enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp¢7enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢>enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢7enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢>enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp¢&enc_past/conv1d/BiasAdd/ReadVariableOp¢2enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢(enc_past/conv1d_1/BiasAdd/ReadVariableOp¢4enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢(enc_past/conv1d_2/BiasAdd/ReadVariableOp¢4enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp¢(spatial_att/dense/BiasAdd/ReadVariableOp¢*spatial_att/dense/Tensordot/ReadVariableOp¢*spatial_att/dense_1/BiasAdd/ReadVariableOp¢,spatial_att/dense_1/Tensordot/ReadVariableOp¢,spatial_att/dense_2/Tensordot/ReadVariableOp¢,spatial_att/dense_3/Tensordot/ReadVariableOp¢,spatial_att/dense_4/Tensordot/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :v

ExpandDims
ExpandDims	x_forwardExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :|
ExpandDims_1
ExpandDimspos_enc_fwdExpandDims_1/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdp
%enc_past/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¡
!enc_past/conv1d/Conv1D/ExpandDims
ExpandDimsx_past.enc_past/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
2enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp;enc_past_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0i
'enc_past/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ð
#enc_past/conv1d/Conv1D/ExpandDims_1
ExpandDims:enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:00enc_past/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Ý
enc_past/conv1d/Conv1DConv2D*enc_past/conv1d/Conv1D/ExpandDims:output:0,enc_past/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
 
enc_past/conv1d/Conv1D/SqueezeSqueezeenc_past/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
&enc_past/conv1d/BiasAdd/ReadVariableOpReadVariableOp/enc_past_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0±
enc_past/conv1d/BiasAddBiasAdd'enc_past/conv1d/Conv1D/Squeeze:output:0.enc_past/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
enc_past/re_lu/ReluRelu enc_past/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
'enc_past/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÀ
#enc_past/conv1d_1/Conv1D/ExpandDims
ExpandDims!enc_past/re_lu/Relu:activations:00enc_past/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¶
4enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=enc_past_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0k
)enc_past/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ö
%enc_past/conv1d_1/Conv1D/ExpandDims_1
ExpandDims<enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:02enc_past/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ã
enc_past/conv1d_1/Conv1DConv2D,enc_past/conv1d_1/Conv1D/ExpandDims:output:0.enc_past/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
¤
 enc_past/conv1d_1/Conv1D/SqueezeSqueeze!enc_past/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
(enc_past/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp1enc_past_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0·
enc_past/conv1d_1/BiasAddBiasAdd)enc_past/conv1d_1/Conv1D/Squeeze:output:00enc_past/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
enc_past/re_lu/Relu_1Relu"enc_past/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
'enc_past/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÂ
#enc_past/conv1d_2/Conv1D/ExpandDims
ExpandDims#enc_past/re_lu/Relu_1:activations:00enc_past/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¶
4enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=enc_past_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0k
)enc_past/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ö
%enc_past/conv1d_2/Conv1D/ExpandDims_1
ExpandDims<enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:02enc_past/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ã
enc_past/conv1d_2/Conv1DConv2D,enc_past/conv1d_2/Conv1D/ExpandDims:output:0.enc_past/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
¤
 enc_past/conv1d_2/Conv1D/SqueezeSqueeze!enc_past/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
(enc_past/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp1enc_past_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0·
enc_past/conv1d_2/BiasAddBiasAdd)enc_past/conv1d_2/Conv1D/Squeeze:output:00enc_past/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
enc_past/re_lu/Relu_2Relu"enc_past/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  
flatten/ReshapeReshape#enc_past/re_lu/Relu_2:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	À *
dtype0
dense_5/MatMulMatMulflatten/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
re_lu_3/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsre_lu_3/Relu:activations:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
Tile/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"   
      r
TileTileExpandDims_2:output:0Tile/multiples:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
concatConcatV2Tile:output:0pos_enc_pastconcat/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes

:$ *
dtype0`
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       V
dense_6/Tensordot/ShapeShapeconcat:output:0*
T0*
_output_shapes
:a
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_6/Tensordot/transpose	Transposeconcat:output:0!dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$¢
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 f
re_lu_3/Relu_1Reludense_6/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 X
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
concat_1ConcatV2ExpandDims:output:0ExpandDims_1:output:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
*spatial_att/dense/Tensordot/ReadVariableOpReadVariableOp3spatial_att_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0j
 spatial_att/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
 spatial_att/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          b
!spatial_att/dense/Tensordot/ShapeShapeconcat_1:output:0*
T0*
_output_shapes
:k
)spatial_att/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
$spatial_att/dense/Tensordot/GatherV2GatherV2*spatial_att/dense/Tensordot/Shape:output:0)spatial_att/dense/Tensordot/free:output:02spatial_att/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+spatial_att/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&spatial_att/dense/Tensordot/GatherV2_1GatherV2*spatial_att/dense/Tensordot/Shape:output:0)spatial_att/dense/Tensordot/axes:output:04spatial_att/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!spatial_att/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¤
 spatial_att/dense/Tensordot/ProdProd-spatial_att/dense/Tensordot/GatherV2:output:0*spatial_att/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#spatial_att/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ª
"spatial_att/dense/Tensordot/Prod_1Prod/spatial_att/dense/Tensordot/GatherV2_1:output:0,spatial_att/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'spatial_att/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ä
"spatial_att/dense/Tensordot/concatConcatV2)spatial_att/dense/Tensordot/free:output:0)spatial_att/dense/Tensordot/axes:output:00spatial_att/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¯
!spatial_att/dense/Tensordot/stackPack)spatial_att/dense/Tensordot/Prod:output:0+spatial_att/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
%spatial_att/dense/Tensordot/transpose	Transposeconcat_1:output:0+spatial_att/dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÀ
#spatial_att/dense/Tensordot/ReshapeReshape)spatial_att/dense/Tensordot/transpose:y:0*spatial_att/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
"spatial_att/dense/Tensordot/MatMulMatMul,spatial_att/dense/Tensordot/Reshape:output:02spatial_att/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
#spatial_att/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:k
)spatial_att/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ï
$spatial_att/dense/Tensordot/concat_1ConcatV2-spatial_att/dense/Tensordot/GatherV2:output:0,spatial_att/dense/Tensordot/Const_2:output:02spatial_att/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:½
spatial_att/dense/TensordotReshape,spatial_att/dense/Tensordot/MatMul:product:0-spatial_att/dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(spatial_att/dense/BiasAdd/ReadVariableOpReadVariableOp1spatial_att_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
spatial_att/dense/BiasAddBiasAdd$spatial_att/dense/Tensordot:output:00spatial_att/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
spatial_att/re_lu_1/ReluRelu"spatial_att/dense/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
,spatial_att/dense_1/Tensordot/ReadVariableOpReadVariableOp5spatial_att_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0l
"spatial_att/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:w
"spatial_att/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          y
#spatial_att/dense_1/Tensordot/ShapeShape&spatial_att/re_lu_1/Relu:activations:0*
T0*
_output_shapes
:m
+spatial_att/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&spatial_att/dense_1/Tensordot/GatherV2GatherV2,spatial_att/dense_1/Tensordot/Shape:output:0+spatial_att/dense_1/Tensordot/free:output:04spatial_att/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-spatial_att/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(spatial_att/dense_1/Tensordot/GatherV2_1GatherV2,spatial_att/dense_1/Tensordot/Shape:output:0+spatial_att/dense_1/Tensordot/axes:output:06spatial_att/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#spatial_att/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
"spatial_att/dense_1/Tensordot/ProdProd/spatial_att/dense_1/Tensordot/GatherV2:output:0,spatial_att/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%spatial_att/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: °
$spatial_att/dense_1/Tensordot/Prod_1Prod1spatial_att/dense_1/Tensordot/GatherV2_1:output:0.spatial_att/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)spatial_att/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ì
$spatial_att/dense_1/Tensordot/concatConcatV2+spatial_att/dense_1/Tensordot/free:output:0+spatial_att/dense_1/Tensordot/axes:output:02spatial_att/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:µ
#spatial_att/dense_1/Tensordot/stackPack+spatial_att/dense_1/Tensordot/Prod:output:0-spatial_att/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Å
'spatial_att/dense_1/Tensordot/transpose	Transpose&spatial_att/re_lu_1/Relu:activations:0-spatial_att/dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÆ
%spatial_att/dense_1/Tensordot/ReshapeReshape+spatial_att/dense_1/Tensordot/transpose:y:0,spatial_att/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
$spatial_att/dense_1/Tensordot/MatMulMatMul.spatial_att/dense_1/Tensordot/Reshape:output:04spatial_att/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
%spatial_att/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:m
+spatial_att/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
&spatial_att/dense_1/Tensordot/concat_1ConcatV2/spatial_att/dense_1/Tensordot/GatherV2:output:0.spatial_att/dense_1/Tensordot/Const_2:output:04spatial_att/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ã
spatial_att/dense_1/TensordotReshape.spatial_att/dense_1/Tensordot/MatMul:product:0/spatial_att/dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
*spatial_att/dense_1/BiasAdd/ReadVariableOpReadVariableOp3spatial_att_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¼
spatial_att/dense_1/BiasAddBiasAdd&spatial_att/dense_1/Tensordot:output:02spatial_att/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
spatial_att/re_lu_1/Relu_1Relu$spatial_att/dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
,spatial_att/dense_3/Tensordot/ReadVariableOpReadVariableOp5spatial_att_dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0l
"spatial_att/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:w
"spatial_att/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          {
#spatial_att/dense_3/Tensordot/ShapeShape(spatial_att/re_lu_1/Relu_1:activations:0*
T0*
_output_shapes
:m
+spatial_att/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&spatial_att/dense_3/Tensordot/GatherV2GatherV2,spatial_att/dense_3/Tensordot/Shape:output:0+spatial_att/dense_3/Tensordot/free:output:04spatial_att/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-spatial_att/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(spatial_att/dense_3/Tensordot/GatherV2_1GatherV2,spatial_att/dense_3/Tensordot/Shape:output:0+spatial_att/dense_3/Tensordot/axes:output:06spatial_att/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#spatial_att/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
"spatial_att/dense_3/Tensordot/ProdProd/spatial_att/dense_3/Tensordot/GatherV2:output:0,spatial_att/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%spatial_att/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: °
$spatial_att/dense_3/Tensordot/Prod_1Prod1spatial_att/dense_3/Tensordot/GatherV2_1:output:0.spatial_att/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)spatial_att/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ì
$spatial_att/dense_3/Tensordot/concatConcatV2+spatial_att/dense_3/Tensordot/free:output:0+spatial_att/dense_3/Tensordot/axes:output:02spatial_att/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:µ
#spatial_att/dense_3/Tensordot/stackPack+spatial_att/dense_3/Tensordot/Prod:output:0-spatial_att/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ç
'spatial_att/dense_3/Tensordot/transpose	Transpose(spatial_att/re_lu_1/Relu_1:activations:0-spatial_att/dense_3/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÆ
%spatial_att/dense_3/Tensordot/ReshapeReshape+spatial_att/dense_3/Tensordot/transpose:y:0,spatial_att/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
$spatial_att/dense_3/Tensordot/MatMulMatMul.spatial_att/dense_3/Tensordot/Reshape:output:04spatial_att/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
%spatial_att/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:m
+spatial_att/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
&spatial_att/dense_3/Tensordot/concat_1ConcatV2/spatial_att/dense_3/Tensordot/GatherV2:output:0.spatial_att/dense_3/Tensordot/Const_2:output:04spatial_att/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ã
spatial_att/dense_3/TensordotReshape.spatial_att/dense_3/Tensordot/MatMul:product:0/spatial_att/dense_3/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
,spatial_att/dense_2/Tensordot/ReadVariableOpReadVariableOp5spatial_att_dense_2_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0l
"spatial_att/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"spatial_att/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       o
#spatial_att/dense_2/Tensordot/ShapeShapere_lu_3/Relu_1:activations:0*
T0*
_output_shapes
:m
+spatial_att/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&spatial_att/dense_2/Tensordot/GatherV2GatherV2,spatial_att/dense_2/Tensordot/Shape:output:0+spatial_att/dense_2/Tensordot/free:output:04spatial_att/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-spatial_att/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(spatial_att/dense_2/Tensordot/GatherV2_1GatherV2,spatial_att/dense_2/Tensordot/Shape:output:0+spatial_att/dense_2/Tensordot/axes:output:06spatial_att/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#spatial_att/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
"spatial_att/dense_2/Tensordot/ProdProd/spatial_att/dense_2/Tensordot/GatherV2:output:0,spatial_att/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%spatial_att/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: °
$spatial_att/dense_2/Tensordot/Prod_1Prod1spatial_att/dense_2/Tensordot/GatherV2_1:output:0.spatial_att/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)spatial_att/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ì
$spatial_att/dense_2/Tensordot/concatConcatV2+spatial_att/dense_2/Tensordot/free:output:0+spatial_att/dense_2/Tensordot/axes:output:02spatial_att/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:µ
#spatial_att/dense_2/Tensordot/stackPack+spatial_att/dense_2/Tensordot/Prod:output:0-spatial_att/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:·
'spatial_att/dense_2/Tensordot/transpose	Transposere_lu_3/Relu_1:activations:0-spatial_att/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Æ
%spatial_att/dense_2/Tensordot/ReshapeReshape+spatial_att/dense_2/Tensordot/transpose:y:0,spatial_att/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
$spatial_att/dense_2/Tensordot/MatMulMatMul.spatial_att/dense_2/Tensordot/Reshape:output:04spatial_att/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
%spatial_att/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:m
+spatial_att/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
&spatial_att/dense_2/Tensordot/concat_1ConcatV2/spatial_att/dense_2/Tensordot/GatherV2:output:0.spatial_att/dense_2/Tensordot/Const_2:output:04spatial_att/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¿
spatial_att/dense_2/TensordotReshape.spatial_att/dense_2/Tensordot/MatMul:product:0/spatial_att/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
\
spatial_att/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :«
spatial_att/ExpandDims
ExpandDims&spatial_att/dense_2/Tensordot:output:0#spatial_att/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

spatial_att/addAddV2&spatial_att/dense_3/Tensordot:output:0spatial_att/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dr
spatial_att/activation/TanhTanhspatial_att/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d¢
,spatial_att/dense_4/Tensordot/ReadVariableOpReadVariableOp5spatial_att_dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0l
"spatial_att/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:w
"spatial_att/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          r
#spatial_att/dense_4/Tensordot/ShapeShapespatial_att/activation/Tanh:y:0*
T0*
_output_shapes
:m
+spatial_att/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&spatial_att/dense_4/Tensordot/GatherV2GatherV2,spatial_att/dense_4/Tensordot/Shape:output:0+spatial_att/dense_4/Tensordot/free:output:04spatial_att/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-spatial_att/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(spatial_att/dense_4/Tensordot/GatherV2_1GatherV2,spatial_att/dense_4/Tensordot/Shape:output:0+spatial_att/dense_4/Tensordot/axes:output:06spatial_att/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#spatial_att/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
"spatial_att/dense_4/Tensordot/ProdProd/spatial_att/dense_4/Tensordot/GatherV2:output:0,spatial_att/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%spatial_att/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: °
$spatial_att/dense_4/Tensordot/Prod_1Prod1spatial_att/dense_4/Tensordot/GatherV2_1:output:0.spatial_att/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)spatial_att/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ì
$spatial_att/dense_4/Tensordot/concatConcatV2+spatial_att/dense_4/Tensordot/free:output:0+spatial_att/dense_4/Tensordot/axes:output:02spatial_att/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:µ
#spatial_att/dense_4/Tensordot/stackPack+spatial_att/dense_4/Tensordot/Prod:output:0-spatial_att/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¾
'spatial_att/dense_4/Tensordot/transpose	Transposespatial_att/activation/Tanh:y:0-spatial_att/dense_4/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dÆ
%spatial_att/dense_4/Tensordot/ReshapeReshape+spatial_att/dense_4/Tensordot/transpose:y:0,spatial_att/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÆ
$spatial_att/dense_4/Tensordot/MatMulMatMul.spatial_att/dense_4/Tensordot/Reshape:output:04spatial_att/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
%spatial_att/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:m
+spatial_att/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
&spatial_att/dense_4/Tensordot/concat_1ConcatV2/spatial_att/dense_4/Tensordot/GatherV2:output:0.spatial_att/dense_4/Tensordot/Const_2:output:04spatial_att/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ã
spatial_att/dense_4/TensordotReshape.spatial_att/dense_4/Tensordot/MatMul:product:0/spatial_att/dense_4/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
spatial_att/SqueezeSqueeze&spatial_att/dense_4/Tensordot:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*
squeeze_dims

ÿÿÿÿÿÿÿÿÿz
spatial_att/softmax/SoftmaxSoftmaxspatial_att/Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dR
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_3
ExpandDims%spatial_att/softmax/Softmax:softmax:0ExpandDims_3/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dp
mulMulExpandDims_3:output:0ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
du
*enc_forward/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ°
&enc_forward/conv1d_3/Conv1D/ExpandDims
ExpandDimsmul:z:03enc_forward/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
d¼
7enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp@enc_forward_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0n
,enc_forward/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ß
(enc_forward/conv1d_3/Conv1D/ExpandDims_1
ExpandDims?enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:05enc_forward/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
!enc_forward/conv1d_3/Conv1D/ShapeShape/enc_forward/conv1d_3/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:y
/enc_forward/conv1d_3/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1enc_forward/conv1d_3/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1enc_forward/conv1d_3/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)enc_forward/conv1d_3/Conv1D/strided_sliceStridedSlice*enc_forward/conv1d_3/Conv1D/Shape:output:08enc_forward/conv1d_3/Conv1D/strided_slice/stack:output:0:enc_forward/conv1d_3/Conv1D/strided_slice/stack_1:output:0:enc_forward/conv1d_3/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)enc_forward/conv1d_3/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   d      Í
#enc_forward/conv1d_3/Conv1D/ReshapeReshape/enc_forward/conv1d_3/Conv1D/ExpandDims:output:02enc_forward/conv1d_3/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdð
"enc_forward/conv1d_3/Conv1D/Conv2DConv2D,enc_forward/conv1d_3/Conv1D/Reshape:output:01enc_forward/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb *
paddingVALID*
strides

+enc_forward/conv1d_3/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   b       r
'enc_forward/conv1d_3/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"enc_forward/conv1d_3/Conv1D/concatConcatV22enc_forward/conv1d_3/Conv1D/strided_slice:output:04enc_forward/conv1d_3/Conv1D/concat/values_1:output:00enc_forward/conv1d_3/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:È
%enc_forward/conv1d_3/Conv1D/Reshape_1Reshape+enc_forward/conv1d_3/Conv1D/Conv2D:output:0+enc_forward/conv1d_3/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b ¸
#enc_forward/conv1d_3/Conv1D/SqueezeSqueeze.enc_forward/conv1d_3/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
-enc_forward/conv1d_3/squeeze_batch_dims/ShapeShape,enc_forward/conv1d_3/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:
;enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
=enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5enc_forward/conv1d_3/squeeze_batch_dims/strided_sliceStridedSlice6enc_forward/conv1d_3/squeeze_batch_dims/Shape:output:0Denc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack:output:0Fenc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_1:output:0Fenc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
5enc_forward/conv1d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿb       Þ
/enc_forward/conv1d_3/squeeze_batch_dims/ReshapeReshape,enc_forward/conv1d_3/Conv1D/Squeeze:output:0>enc_forward/conv1d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb Â
>enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpGenc_forward_conv1d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ò
/enc_forward/conv1d_3/squeeze_batch_dims/BiasAddBiasAdd8enc_forward/conv1d_3/squeeze_batch_dims/Reshape:output:0Fenc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb 
7enc_forward/conv1d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"b       ~
3enc_forward/conv1d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¨
.enc_forward/conv1d_3/squeeze_batch_dims/concatConcatV2>enc_forward/conv1d_3/squeeze_batch_dims/strided_slice:output:0@enc_forward/conv1d_3/squeeze_batch_dims/concat/values_1:output:0<enc_forward/conv1d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:é
1enc_forward/conv1d_3/squeeze_batch_dims/Reshape_1Reshape8enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd:output:07enc_forward/conv1d_3/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b 
enc_forward/re_lu_2/ReluRelu:enc_forward/conv1d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b u
*enc_forward/conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÏ
&enc_forward/conv1d_4/Conv1D/ExpandDims
ExpandDims&enc_forward/re_lu_2/Relu:activations:03enc_forward/conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b ¼
7enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp@enc_forward_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0n
,enc_forward/conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ß
(enc_forward/conv1d_4/Conv1D/ExpandDims_1
ExpandDims?enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:05enc_forward/conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  
!enc_forward/conv1d_4/Conv1D/ShapeShape/enc_forward/conv1d_4/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:y
/enc_forward/conv1d_4/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1enc_forward/conv1d_4/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1enc_forward/conv1d_4/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)enc_forward/conv1d_4/Conv1D/strided_sliceStridedSlice*enc_forward/conv1d_4/Conv1D/Shape:output:08enc_forward/conv1d_4/Conv1D/strided_slice/stack:output:0:enc_forward/conv1d_4/Conv1D/strided_slice/stack_1:output:0:enc_forward/conv1d_4/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)enc_forward/conv1d_4/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   b       Í
#enc_forward/conv1d_4/Conv1D/ReshapeReshape/enc_forward/conv1d_4/Conv1D/ExpandDims:output:02enc_forward/conv1d_4/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb ð
"enc_forward/conv1d_4/Conv1D/Conv2DConv2D,enc_forward/conv1d_4/Conv1D/Reshape:output:01enc_forward/conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` *
paddingVALID*
strides

+enc_forward/conv1d_4/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   `       r
'enc_forward/conv1d_4/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"enc_forward/conv1d_4/Conv1D/concatConcatV22enc_forward/conv1d_4/Conv1D/strided_slice:output:04enc_forward/conv1d_4/Conv1D/concat/values_1:output:00enc_forward/conv1d_4/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:È
%enc_forward/conv1d_4/Conv1D/Reshape_1Reshape+enc_forward/conv1d_4/Conv1D/Conv2D:output:0+enc_forward/conv1d_4/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` ¸
#enc_forward/conv1d_4/Conv1D/SqueezeSqueeze.enc_forward/conv1d_4/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
-enc_forward/conv1d_4/squeeze_batch_dims/ShapeShape,enc_forward/conv1d_4/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:
;enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
=enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5enc_forward/conv1d_4/squeeze_batch_dims/strided_sliceStridedSlice6enc_forward/conv1d_4/squeeze_batch_dims/Shape:output:0Denc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack:output:0Fenc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_1:output:0Fenc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
5enc_forward/conv1d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ`       Þ
/enc_forward/conv1d_4/squeeze_batch_dims/ReshapeReshape,enc_forward/conv1d_4/Conv1D/Squeeze:output:0>enc_forward/conv1d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` Â
>enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpGenc_forward_conv1d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ò
/enc_forward/conv1d_4/squeeze_batch_dims/BiasAddBiasAdd8enc_forward/conv1d_4/squeeze_batch_dims/Reshape:output:0Fenc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` 
7enc_forward/conv1d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"`       ~
3enc_forward/conv1d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¨
.enc_forward/conv1d_4/squeeze_batch_dims/concatConcatV2>enc_forward/conv1d_4/squeeze_batch_dims/strided_slice:output:0@enc_forward/conv1d_4/squeeze_batch_dims/concat/values_1:output:0<enc_forward/conv1d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:é
1enc_forward/conv1d_4/squeeze_batch_dims/Reshape_1Reshape8enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd:output:07enc_forward/conv1d_4/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
enc_forward/re_lu_2/Relu_1Relu:enc_forward/conv1d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` u
*enc_forward/conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÑ
&enc_forward/conv1d_5/Conv1D/ExpandDims
ExpandDims(enc_forward/re_lu_2/Relu_1:activations:03enc_forward/conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` ¼
7enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp@enc_forward_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0n
,enc_forward/conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ß
(enc_forward/conv1d_5/Conv1D/ExpandDims_1
ExpandDims?enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:05enc_forward/conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  
!enc_forward/conv1d_5/Conv1D/ShapeShape/enc_forward/conv1d_5/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:y
/enc_forward/conv1d_5/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1enc_forward/conv1d_5/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ{
1enc_forward/conv1d_5/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
)enc_forward/conv1d_5/Conv1D/strided_sliceStridedSlice*enc_forward/conv1d_5/Conv1D/Shape:output:08enc_forward/conv1d_5/Conv1D/strided_slice/stack:output:0:enc_forward/conv1d_5/Conv1D/strided_slice/stack_1:output:0:enc_forward/conv1d_5/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
)enc_forward/conv1d_5/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   `       Í
#enc_forward/conv1d_5/Conv1D/ReshapeReshape/enc_forward/conv1d_5/Conv1D/ExpandDims:output:02enc_forward/conv1d_5/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` ð
"enc_forward/conv1d_5/Conv1D/Conv2DConv2D,enc_forward/conv1d_5/Conv1D/Reshape:output:01enc_forward/conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ *
paddingVALID*
strides

+enc_forward/conv1d_5/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   ^       r
'enc_forward/conv1d_5/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿø
"enc_forward/conv1d_5/Conv1D/concatConcatV22enc_forward/conv1d_5/Conv1D/strided_slice:output:04enc_forward/conv1d_5/Conv1D/concat/values_1:output:00enc_forward/conv1d_5/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:È
%enc_forward/conv1d_5/Conv1D/Reshape_1Reshape+enc_forward/conv1d_5/Conv1D/Conv2D:output:0+enc_forward/conv1d_5/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
^ ¸
#enc_forward/conv1d_5/Conv1D/SqueezeSqueeze.enc_forward/conv1d_5/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
-enc_forward/conv1d_5/squeeze_batch_dims/ShapeShape,enc_forward/conv1d_5/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:
;enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
=enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5enc_forward/conv1d_5/squeeze_batch_dims/strided_sliceStridedSlice6enc_forward/conv1d_5/squeeze_batch_dims/Shape:output:0Denc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack:output:0Fenc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_1:output:0Fenc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
5enc_forward/conv1d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ^       Þ
/enc_forward/conv1d_5/squeeze_batch_dims/ReshapeReshape,enc_forward/conv1d_5/Conv1D/Squeeze:output:0>enc_forward/conv1d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ Â
>enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpGenc_forward_conv1d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ò
/enc_forward/conv1d_5/squeeze_batch_dims/BiasAddBiasAdd8enc_forward/conv1d_5/squeeze_batch_dims/Reshape:output:0Fenc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ 
7enc_forward/conv1d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"^       ~
3enc_forward/conv1d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¨
.enc_forward/conv1d_5/squeeze_batch_dims/concatConcatV2>enc_forward/conv1d_5/squeeze_batch_dims/strided_slice:output:0@enc_forward/conv1d_5/squeeze_batch_dims/concat/values_1:output:0<enc_forward/conv1d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:é
1enc_forward/conv1d_5/squeeze_batch_dims/Reshape_1Reshape8enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd:output:07enc_forward/conv1d_5/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ 
enc_forward/re_lu_2/Relu_2Relu:enc_forward/conv1d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ
   À  
ReshapeReshape(enc_forward/re_lu_2/Relu_2:activations:0Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes
:	À *
dtype0`
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       W
dense_7/Tensordot/ShapeShapeReshape:output:0*
T0*
_output_shapes
:a
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_7/Tensordot/transpose	TransposeReshape:output:0!dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À¢
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 f
re_lu_3/Relu_2Reludense_7/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 X
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ§
concat_2ConcatV2re_lu_3/Relu_1:activations:0re_lu_3/Relu_2:activations:0concat_2/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0`
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       X
dense_8/Tensordot/ShapeShapeconcat_2:output:0*
T0*
_output_shapes
:a
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_8/Tensordot/transpose	Transposeconcat_2:output:0!dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@¢
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 f
re_lu_3/Relu_3Reludense_8/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_9/Tensordot/ShapeShapere_lu_3/Relu_3:activations:0*
T0*
_output_shapes
:a
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_9/Tensordot/transpose	Transposere_lu_3/Relu_3:activations:0!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ¢
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
activation_1/TanhTanhdense_9/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿÿÿÿÿj
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         è
strided_sliceStridedSlicex_paststrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_maskq
addAddV2activation_1/Tanh:y:0strided_slice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ë

NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp8^enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp?^enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp8^enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp?^enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp8^enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp?^enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp'^enc_past/conv1d/BiasAdd/ReadVariableOp3^enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp)^enc_past/conv1d_1/BiasAdd/ReadVariableOp5^enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp)^enc_past/conv1d_2/BiasAdd/ReadVariableOp5^enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp)^spatial_att/dense/BiasAdd/ReadVariableOp+^spatial_att/dense/Tensordot/ReadVariableOp+^spatial_att/dense_1/BiasAdd/ReadVariableOp-^spatial_att/dense_1/Tensordot/ReadVariableOp-^spatial_att/dense_2/Tensordot/ReadVariableOp-^spatial_att/dense_3/Tensordot/ReadVariableOp-^spatial_att/dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*«
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp2r
7enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp7enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2
>enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp>enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2r
7enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp7enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2
>enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp>enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2r
7enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp7enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2
>enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp>enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2P
&enc_past/conv1d/BiasAdd/ReadVariableOp&enc_past/conv1d/BiasAdd/ReadVariableOp2h
2enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2T
(enc_past/conv1d_1/BiasAdd/ReadVariableOp(enc_past/conv1d_1/BiasAdd/ReadVariableOp2l
4enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp4enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2T
(enc_past/conv1d_2/BiasAdd/ReadVariableOp(enc_past/conv1d_2/BiasAdd/ReadVariableOp2l
4enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp4enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2T
(spatial_att/dense/BiasAdd/ReadVariableOp(spatial_att/dense/BiasAdd/ReadVariableOp2X
*spatial_att/dense/Tensordot/ReadVariableOp*spatial_att/dense/Tensordot/ReadVariableOp2X
*spatial_att/dense_1/BiasAdd/ReadVariableOp*spatial_att/dense_1/BiasAdd/ReadVariableOp2\
,spatial_att/dense_1/Tensordot/ReadVariableOp,spatial_att/dense_1/Tensordot/ReadVariableOp2\
,spatial_att/dense_2/Tensordot/ReadVariableOp,spatial_att/dense_2/Tensordot/ReadVariableOp2\
,spatial_att/dense_3/Tensordot/ReadVariableOp,spatial_att/dense_3/Tensordot/ReadVariableOp2\
,spatial_att/dense_4/Tensordot/ReadVariableOp,spatial_att/dense_4/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namex_past:VR
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#
_user_specified_name	x_forward:YU
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&
_user_specified_namepos_enc_past:XT
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namepos_enc_fwd
Ä

³
-__inference_spatial_att_layer_call_fn_1631291	
query
feature
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4: 
	unknown_5:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallqueryfeatureunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*)
_read_only_resource_inputs
	*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_spatial_att_layer_call_and_return_conditional_losses_1629490s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
 :ÿÿÿÿÿÿÿÿÿd: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 

_user_specified_namequery:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	feature
Û

)__inference_dense_9_layer_call_fn_1631970

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1629630s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
Ý)
º
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1632219

inputsA
+conv1d_expanddims_1_readvariableop_resource: @
2squeeze_batch_dims_biasadd_readvariableop_resource: 
identity¢"Conv1D/ExpandDims_1/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
d
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: V
Conv1D/ShapeShapeConv1D/ExpandDims:output:0*
T0*
_output_shapes
:d
Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿf
Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Conv1D/strided_sliceStridedSliceConv1D/Shape:output:0#Conv1D/strided_slice/stack:output:0%Conv1D/strided_slice/stack_1:output:0%Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   d      
Conv1D/ReshapeReshapeConv1D/ExpandDims:output:0Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd±
Conv1D/Conv2DConv2DConv1D/Reshape:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb *
paddingVALID*
strides
k
Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   b       ]
Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
Conv1D/concatConcatV2Conv1D/strided_slice:output:0Conv1D/concat/values_1:output:0Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:
Conv1D/Reshape_1ReshapeConv1D/Conv2D:output:0Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b 
Conv1D/SqueezeSqueezeConv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *
squeeze_dims

ýÿÿÿÿÿÿÿÿ_
squeeze_batch_dims/ShapeShapeConv1D/Squeeze:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿr
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿb       
squeeze_batch_dims/ReshapeReshapeConv1D/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb 
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb s
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"b       i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÔ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ª
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b |
IdentityIdentity%squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b 
NoOpNoOp#^Conv1D/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
d: : 2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
 
_user_specified_nameinputs
Û

)__inference_dense_8_layer_call_fn_1631931

inputs
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1629593s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@
 
_user_specified_nameinputs
Äu
õ
#__inference__traced_restore_1632521
file_prefix;
(assignvariableop_conv_att_dense_5_kernel:	À 6
(assignvariableop_1_conv_att_dense_5_bias: <
*assignvariableop_2_conv_att_dense_6_kernel:$ 6
(assignvariableop_3_conv_att_dense_6_bias: =
*assignvariableop_4_conv_att_dense_7_kernel:	À 6
(assignvariableop_5_conv_att_dense_7_bias: <
*assignvariableop_6_conv_att_dense_8_kernel:@ 6
(assignvariableop_7_conv_att_dense_8_bias: <
*assignvariableop_8_conv_att_dense_9_kernel: 6
(assignvariableop_9_conv_att_dense_9_bias:I
3assignvariableop_10_conv_att_enc_past_conv1d_kernel: ?
1assignvariableop_11_conv_att_enc_past_conv1d_bias: K
5assignvariableop_12_conv_att_enc_past_conv1d_1_kernel:  A
3assignvariableop_13_conv_att_enc_past_conv1d_1_bias: K
5assignvariableop_14_conv_att_enc_past_conv1d_2_kernel:  A
3assignvariableop_15_conv_att_enc_past_conv1d_2_bias: G
5assignvariableop_16_conv_att_spatial_att_dense_kernel:A
3assignvariableop_17_conv_att_spatial_att_dense_bias:I
7assignvariableop_18_conv_att_spatial_att_dense_1_kernel:C
5assignvariableop_19_conv_att_spatial_att_dense_1_bias:I
7assignvariableop_20_conv_att_spatial_att_dense_2_kernel: I
7assignvariableop_21_conv_att_spatial_att_dense_3_kernel:I
7assignvariableop_22_conv_att_spatial_att_dense_4_kernel:N
8assignvariableop_23_conv_att_enc_forward_conv1d_3_kernel: D
6assignvariableop_24_conv_att_enc_forward_conv1d_3_bias: N
8assignvariableop_25_conv_att_enc_forward_conv1d_4_kernel:  D
6assignvariableop_26_conv_att_enc_forward_conv1d_4_bias: N
8assignvariableop_27_conv_att_enc_forward_conv1d_5_kernel:  D
6assignvariableop_28_conv_att_enc_forward_conv1d_5_bias: 
identity_30¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¨

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Î	
valueÄ	BÁ	B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense3/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense4/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense4/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense5/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B µ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp(assignvariableop_conv_att_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp(assignvariableop_1_conv_att_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp*assignvariableop_2_conv_att_dense_6_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp(assignvariableop_3_conv_att_dense_6_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp*assignvariableop_4_conv_att_dense_7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp(assignvariableop_5_conv_att_dense_7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp*assignvariableop_6_conv_att_dense_8_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp(assignvariableop_7_conv_att_dense_8_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp*assignvariableop_8_conv_att_dense_9_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp(assignvariableop_9_conv_att_dense_9_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_10AssignVariableOp3assignvariableop_10_conv_att_enc_past_conv1d_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_11AssignVariableOp1assignvariableop_11_conv_att_enc_past_conv1d_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_12AssignVariableOp5assignvariableop_12_conv_att_enc_past_conv1d_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_13AssignVariableOp3assignvariableop_13_conv_att_enc_past_conv1d_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_14AssignVariableOp5assignvariableop_14_conv_att_enc_past_conv1d_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_15AssignVariableOp3assignvariableop_15_conv_att_enc_past_conv1d_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_16AssignVariableOp5assignvariableop_16_conv_att_spatial_att_dense_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_17AssignVariableOp3assignvariableop_17_conv_att_spatial_att_dense_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_18AssignVariableOp7assignvariableop_18_conv_att_spatial_att_dense_1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_19AssignVariableOp5assignvariableop_19_conv_att_spatial_att_dense_1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_20AssignVariableOp7assignvariableop_20_conv_att_spatial_att_dense_2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_21AssignVariableOp7assignvariableop_21_conv_att_spatial_att_dense_3_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_22AssignVariableOp7assignvariableop_22_conv_att_spatial_att_dense_4_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_23AssignVariableOp8assignvariableop_23_conv_att_enc_forward_conv1d_3_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_24AssignVariableOp6assignvariableop_24_conv_att_enc_forward_conv1d_3_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_25AssignVariableOp8assignvariableop_25_conv_att_enc_forward_conv1d_4_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_26AssignVariableOp6assignvariableop_26_conv_att_enc_forward_conv1d_4_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_27AssignVariableOp8assignvariableop_27_conv_att_enc_forward_conv1d_5_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_28AssignVariableOp6assignvariableop_28_conv_att_enc_forward_conv1d_5_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Í
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_30IdentityIdentity_29:output:0^NoOp_1*
T0*
_output_shapes
: º
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_30Identity_30:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¦
E
)__inference_re_lu_3_layer_call_fn_1632010

inputs
identity¸
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1629304`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Î

)__inference_dense_5_layer_call_fn_1631834

inputs
unknown:	À 
	unknown_0: 
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_1629293o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
Û

"__inference__wrapped_model_1628643

args_0

args_1

args_2

args_3Z
Dconv_att_enc_past_conv1d_conv1d_expanddims_1_readvariableop_resource: F
8conv_att_enc_past_conv1d_biasadd_readvariableop_resource: \
Fconv_att_enc_past_conv1d_1_conv1d_expanddims_1_readvariableop_resource:  H
:conv_att_enc_past_conv1d_1_biasadd_readvariableop_resource: \
Fconv_att_enc_past_conv1d_2_conv1d_expanddims_1_readvariableop_resource:  H
:conv_att_enc_past_conv1d_2_biasadd_readvariableop_resource: B
/conv_att_dense_5_matmul_readvariableop_resource:	À >
0conv_att_dense_5_biasadd_readvariableop_resource: D
2conv_att_dense_6_tensordot_readvariableop_resource:$ >
0conv_att_dense_6_biasadd_readvariableop_resource: N
<conv_att_spatial_att_dense_tensordot_readvariableop_resource:H
:conv_att_spatial_att_dense_biasadd_readvariableop_resource:P
>conv_att_spatial_att_dense_1_tensordot_readvariableop_resource:J
<conv_att_spatial_att_dense_1_biasadd_readvariableop_resource:P
>conv_att_spatial_att_dense_3_tensordot_readvariableop_resource:P
>conv_att_spatial_att_dense_2_tensordot_readvariableop_resource: P
>conv_att_spatial_att_dense_4_tensordot_readvariableop_resource:_
Iconv_att_enc_forward_conv1d_3_conv1d_expanddims_1_readvariableop_resource: ^
Pconv_att_enc_forward_conv1d_3_squeeze_batch_dims_biasadd_readvariableop_resource: _
Iconv_att_enc_forward_conv1d_4_conv1d_expanddims_1_readvariableop_resource:  ^
Pconv_att_enc_forward_conv1d_4_squeeze_batch_dims_biasadd_readvariableop_resource: _
Iconv_att_enc_forward_conv1d_5_conv1d_expanddims_1_readvariableop_resource:  ^
Pconv_att_enc_forward_conv1d_5_squeeze_batch_dims_biasadd_readvariableop_resource: E
2conv_att_dense_7_tensordot_readvariableop_resource:	À >
0conv_att_dense_7_biasadd_readvariableop_resource: D
2conv_att_dense_8_tensordot_readvariableop_resource:@ >
0conv_att_dense_8_biasadd_readvariableop_resource: D
2conv_att_dense_9_tensordot_readvariableop_resource: >
0conv_att_dense_9_biasadd_readvariableop_resource:
identity¢'conv_att/dense_5/BiasAdd/ReadVariableOp¢&conv_att/dense_5/MatMul/ReadVariableOp¢'conv_att/dense_6/BiasAdd/ReadVariableOp¢)conv_att/dense_6/Tensordot/ReadVariableOp¢'conv_att/dense_7/BiasAdd/ReadVariableOp¢)conv_att/dense_7/Tensordot/ReadVariableOp¢'conv_att/dense_8/BiasAdd/ReadVariableOp¢)conv_att/dense_8/Tensordot/ReadVariableOp¢'conv_att/dense_9/BiasAdd/ReadVariableOp¢)conv_att/dense_9/Tensordot/ReadVariableOp¢@conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp¢Gconv_att/enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp¢@conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢Gconv_att/enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp¢@conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢Gconv_att/enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp¢/conv_att/enc_past/conv1d/BiasAdd/ReadVariableOp¢;conv_att/enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢1conv_att/enc_past/conv1d_1/BiasAdd/ReadVariableOp¢=conv_att/enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢1conv_att/enc_past/conv1d_2/BiasAdd/ReadVariableOp¢=conv_att/enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp¢1conv_att/spatial_att/dense/BiasAdd/ReadVariableOp¢3conv_att/spatial_att/dense/Tensordot/ReadVariableOp¢3conv_att/spatial_att/dense_1/BiasAdd/ReadVariableOp¢5conv_att/spatial_att/dense_1/Tensordot/ReadVariableOp¢5conv_att/spatial_att/dense_2/Tensordot/ReadVariableOp¢5conv_att/spatial_att/dense_3/Tensordot/ReadVariableOp¢5conv_att/spatial_att/dense_4/Tensordot/ReadVariableOpY
conv_att/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
conv_att/ExpandDims
ExpandDimsargs_1 conv_att/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
conv_att/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
conv_att/ExpandDims_1
ExpandDimsargs_3"conv_att/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
.conv_att/enc_past/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ³
*conv_att/enc_past/conv1d/Conv1D/ExpandDims
ExpandDimsargs_07conv_att/enc_past/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
;conv_att/enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpDconv_att_enc_past_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0r
0conv_att/enc_past/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ë
,conv_att/enc_past/conv1d/Conv1D/ExpandDims_1
ExpandDimsCconv_att/enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:09conv_att/enc_past/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ø
conv_att/enc_past/conv1d/Conv1DConv2D3conv_att/enc_past/conv1d/Conv1D/ExpandDims:output:05conv_att/enc_past/conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
²
'conv_att/enc_past/conv1d/Conv1D/SqueezeSqueeze(conv_att/enc_past/conv1d/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ¤
/conv_att/enc_past/conv1d/BiasAdd/ReadVariableOpReadVariableOp8conv_att_enc_past_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ì
 conv_att/enc_past/conv1d/BiasAddBiasAdd0conv_att/enc_past/conv1d/Conv1D/Squeeze:output:07conv_att/enc_past/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_att/enc_past/re_lu/ReluRelu)conv_att/enc_past/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
0conv_att/enc_past/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÛ
,conv_att/enc_past/conv1d_1/Conv1D/ExpandDims
ExpandDims*conv_att/enc_past/re_lu/Relu:activations:09conv_att/enc_past/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
=conv_att/enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpFconv_att_enc_past_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0t
2conv_att/enc_past/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ñ
.conv_att/enc_past/conv1d_1/Conv1D/ExpandDims_1
ExpandDimsEconv_att/enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0;conv_att/enc_past/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  þ
!conv_att/enc_past/conv1d_1/Conv1DConv2D5conv_att/enc_past/conv1d_1/Conv1D/ExpandDims:output:07conv_att/enc_past/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
¶
)conv_att/enc_past/conv1d_1/Conv1D/SqueezeSqueeze*conv_att/enc_past/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ¨
1conv_att/enc_past/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp:conv_att_enc_past_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ò
"conv_att/enc_past/conv1d_1/BiasAddBiasAdd2conv_att/enc_past/conv1d_1/Conv1D/Squeeze:output:09conv_att/enc_past/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_att/enc_past/re_lu/Relu_1Relu+conv_att/enc_past/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
0conv_att/enc_past/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÝ
,conv_att/enc_past/conv1d_2/Conv1D/ExpandDims
ExpandDims,conv_att/enc_past/re_lu/Relu_1:activations:09conv_att/enc_past/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ È
=conv_att/enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpFconv_att_enc_past_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0t
2conv_att/enc_past/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ñ
.conv_att/enc_past/conv1d_2/Conv1D/ExpandDims_1
ExpandDimsEconv_att/enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0;conv_att/enc_past/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  þ
!conv_att/enc_past/conv1d_2/Conv1DConv2D5conv_att/enc_past/conv1d_2/Conv1D/ExpandDims:output:07conv_att/enc_past/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
¶
)conv_att/enc_past/conv1d_2/Conv1D/SqueezeSqueeze*conv_att/enc_past/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ¨
1conv_att/enc_past/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp:conv_att_enc_past_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ò
"conv_att/enc_past/conv1d_2/BiasAddBiasAdd2conv_att/enc_past/conv1d_2/Conv1D/Squeeze:output:09conv_att/enc_past/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv_att/enc_past/re_lu/Relu_2Relu+conv_att/enc_past/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
conv_att/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  ¥
conv_att/flatten/ReshapeReshape,conv_att/enc_past/re_lu/Relu_2:activations:0conv_att/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
&conv_att/dense_5/MatMul/ReadVariableOpReadVariableOp/conv_att_dense_5_matmul_readvariableop_resource*
_output_shapes
:	À *
dtype0¦
conv_att/dense_5/MatMulMatMul!conv_att/flatten/Reshape:output:0.conv_att/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'conv_att/dense_5/BiasAdd/ReadVariableOpReadVariableOp0conv_att_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
conv_att/dense_5/BiasAddBiasAdd!conv_att/dense_5/MatMul:product:0/conv_att/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
conv_att/re_lu_3/ReluRelu!conv_att/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
conv_att/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :¢
conv_att/ExpandDims_2
ExpandDims#conv_att/re_lu_3/Relu:activations:0"conv_att/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
conv_att/Tile/multiplesConst*
_output_shapes
:*
dtype0*!
valueB"   
      
conv_att/TileTileconv_att/ExpandDims_2:output:0 conv_att/Tile/multiples:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 _
conv_att/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
conv_att/concatConcatV2conv_att/Tile:output:0args_2conv_att/concat/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
)conv_att/dense_6/Tensordot/ReadVariableOpReadVariableOp2conv_att_dense_6_tensordot_readvariableop_resource*
_output_shapes

:$ *
dtype0i
conv_att/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
conv_att/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       h
 conv_att/dense_6/Tensordot/ShapeShapeconv_att/concat:output:0*
T0*
_output_shapes
:j
(conv_att/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#conv_att/dense_6/Tensordot/GatherV2GatherV2)conv_att/dense_6/Tensordot/Shape:output:0(conv_att/dense_6/Tensordot/free:output:01conv_att/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*conv_att/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%conv_att/dense_6/Tensordot/GatherV2_1GatherV2)conv_att/dense_6/Tensordot/Shape:output:0(conv_att/dense_6/Tensordot/axes:output:03conv_att/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 conv_att/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
conv_att/dense_6/Tensordot/ProdProd,conv_att/dense_6/Tensordot/GatherV2:output:0)conv_att/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"conv_att/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!conv_att/dense_6/Tensordot/Prod_1Prod.conv_att/dense_6/Tensordot/GatherV2_1:output:0+conv_att/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&conv_att/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!conv_att/dense_6/Tensordot/concatConcatV2(conv_att/dense_6/Tensordot/free:output:0(conv_att/dense_6/Tensordot/axes:output:0/conv_att/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 conv_att/dense_6/Tensordot/stackPack(conv_att/dense_6/Tensordot/Prod:output:0*conv_att/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:­
$conv_att/dense_6/Tensordot/transpose	Transposeconv_att/concat:output:0*conv_att/dense_6/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$½
"conv_att/dense_6/Tensordot/ReshapeReshape(conv_att/dense_6/Tensordot/transpose:y:0)conv_att/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
!conv_att/dense_6/Tensordot/MatMulMatMul+conv_att/dense_6/Tensordot/Reshape:output:01conv_att/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
"conv_att/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: j
(conv_att/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#conv_att/dense_6/Tensordot/concat_1ConcatV2,conv_att/dense_6/Tensordot/GatherV2:output:0+conv_att/dense_6/Tensordot/Const_2:output:01conv_att/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¶
conv_att/dense_6/TensordotReshape+conv_att/dense_6/Tensordot/MatMul:product:0,conv_att/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
'conv_att/dense_6/BiasAdd/ReadVariableOpReadVariableOp0conv_att_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¯
conv_att/dense_6/BiasAddBiasAdd#conv_att/dense_6/Tensordot:output:0/conv_att/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 x
conv_att/re_lu_3/Relu_1Relu!conv_att/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 a
conv_att/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¿
conv_att/concat_1ConcatV2conv_att/ExpandDims:output:0conv_att/ExpandDims_1:output:0conv_att/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd°
3conv_att/spatial_att/dense/Tensordot/ReadVariableOpReadVariableOp<conv_att_spatial_att_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0s
)conv_att/spatial_att/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
)conv_att/spatial_att/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          t
*conv_att/spatial_att/dense/Tensordot/ShapeShapeconv_att/concat_1:output:0*
T0*
_output_shapes
:t
2conv_att/spatial_att/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : §
-conv_att/spatial_att/dense/Tensordot/GatherV2GatherV23conv_att/spatial_att/dense/Tensordot/Shape:output:02conv_att/spatial_att/dense/Tensordot/free:output:0;conv_att/spatial_att/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4conv_att/spatial_att/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : «
/conv_att/spatial_att/dense/Tensordot/GatherV2_1GatherV23conv_att/spatial_att/dense/Tensordot/Shape:output:02conv_att/spatial_att/dense/Tensordot/axes:output:0=conv_att/spatial_att/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*conv_att/spatial_att/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¿
)conv_att/spatial_att/dense/Tensordot/ProdProd6conv_att/spatial_att/dense/Tensordot/GatherV2:output:03conv_att/spatial_att/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,conv_att/spatial_att/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Å
+conv_att/spatial_att/dense/Tensordot/Prod_1Prod8conv_att/spatial_att/dense/Tensordot/GatherV2_1:output:05conv_att/spatial_att/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0conv_att/spatial_att/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+conv_att/spatial_att/dense/Tensordot/concatConcatV22conv_att/spatial_att/dense/Tensordot/free:output:02conv_att/spatial_att/dense/Tensordot/axes:output:09conv_att/spatial_att/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ê
*conv_att/spatial_att/dense/Tensordot/stackPack2conv_att/spatial_att/dense/Tensordot/Prod:output:04conv_att/spatial_att/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ç
.conv_att/spatial_att/dense/Tensordot/transpose	Transposeconv_att/concat_1:output:04conv_att/spatial_att/dense/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÛ
,conv_att/spatial_att/dense/Tensordot/ReshapeReshape2conv_att/spatial_att/dense/Tensordot/transpose:y:03conv_att/spatial_att/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÛ
+conv_att/spatial_att/dense/Tensordot/MatMulMatMul5conv_att/spatial_att/dense/Tensordot/Reshape:output:0;conv_att/spatial_att/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
,conv_att/spatial_att/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:t
2conv_att/spatial_att/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-conv_att/spatial_att/dense/Tensordot/concat_1ConcatV26conv_att/spatial_att/dense/Tensordot/GatherV2:output:05conv_att/spatial_att/dense/Tensordot/Const_2:output:0;conv_att/spatial_att/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ø
$conv_att/spatial_att/dense/TensordotReshape5conv_att/spatial_att/dense/Tensordot/MatMul:product:06conv_att/spatial_att/dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¨
1conv_att/spatial_att/dense/BiasAdd/ReadVariableOpReadVariableOp:conv_att_spatial_att_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ñ
"conv_att/spatial_att/dense/BiasAddBiasAdd-conv_att/spatial_att/dense/Tensordot:output:09conv_att/spatial_att/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!conv_att/spatial_att/re_lu_1/ReluRelu+conv_att/spatial_att/dense/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd´
5conv_att/spatial_att/dense_1/Tensordot/ReadVariableOpReadVariableOp>conv_att_spatial_att_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0u
+conv_att/spatial_att/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
+conv_att/spatial_att/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          
,conv_att/spatial_att/dense_1/Tensordot/ShapeShape/conv_att/spatial_att/re_lu_1/Relu:activations:0*
T0*
_output_shapes
:v
4conv_att/spatial_att/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¯
/conv_att/spatial_att/dense_1/Tensordot/GatherV2GatherV25conv_att/spatial_att/dense_1/Tensordot/Shape:output:04conv_att/spatial_att/dense_1/Tensordot/free:output:0=conv_att/spatial_att/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6conv_att/spatial_att/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
1conv_att/spatial_att/dense_1/Tensordot/GatherV2_1GatherV25conv_att/spatial_att/dense_1/Tensordot/Shape:output:04conv_att/spatial_att/dense_1/Tensordot/axes:output:0?conv_att/spatial_att/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,conv_att/spatial_att/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
+conv_att/spatial_att/dense_1/Tensordot/ProdProd8conv_att/spatial_att/dense_1/Tensordot/GatherV2:output:05conv_att/spatial_att/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.conv_att/spatial_att/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ë
-conv_att/spatial_att/dense_1/Tensordot/Prod_1Prod:conv_att/spatial_att/dense_1/Tensordot/GatherV2_1:output:07conv_att/spatial_att/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2conv_att/spatial_att/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-conv_att/spatial_att/dense_1/Tensordot/concatConcatV24conv_att/spatial_att/dense_1/Tensordot/free:output:04conv_att/spatial_att/dense_1/Tensordot/axes:output:0;conv_att/spatial_att/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ð
,conv_att/spatial_att/dense_1/Tensordot/stackPack4conv_att/spatial_att/dense_1/Tensordot/Prod:output:06conv_att/spatial_att/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:à
0conv_att/spatial_att/dense_1/Tensordot/transpose	Transpose/conv_att/spatial_att/re_lu_1/Relu:activations:06conv_att/spatial_att/dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdá
.conv_att/spatial_att/dense_1/Tensordot/ReshapeReshape4conv_att/spatial_att/dense_1/Tensordot/transpose:y:05conv_att/spatial_att/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
-conv_att/spatial_att/dense_1/Tensordot/MatMulMatMul7conv_att/spatial_att/dense_1/Tensordot/Reshape:output:0=conv_att/spatial_att/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
.conv_att/spatial_att/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:v
4conv_att/spatial_att/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
/conv_att/spatial_att/dense_1/Tensordot/concat_1ConcatV28conv_att/spatial_att/dense_1/Tensordot/GatherV2:output:07conv_att/spatial_att/dense_1/Tensordot/Const_2:output:0=conv_att/spatial_att/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Þ
&conv_att/spatial_att/dense_1/TensordotReshape7conv_att/spatial_att/dense_1/Tensordot/MatMul:product:08conv_att/spatial_att/dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¬
3conv_att/spatial_att/dense_1/BiasAdd/ReadVariableOpReadVariableOp<conv_att_spatial_att_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0×
$conv_att/spatial_att/dense_1/BiasAddBiasAdd/conv_att/spatial_att/dense_1/Tensordot:output:0;conv_att/spatial_att/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#conv_att/spatial_att/re_lu_1/Relu_1Relu-conv_att/spatial_att/dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd´
5conv_att/spatial_att/dense_3/Tensordot/ReadVariableOpReadVariableOp>conv_att_spatial_att_dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0u
+conv_att/spatial_att/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
+conv_att/spatial_att/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          
,conv_att/spatial_att/dense_3/Tensordot/ShapeShape1conv_att/spatial_att/re_lu_1/Relu_1:activations:0*
T0*
_output_shapes
:v
4conv_att/spatial_att/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¯
/conv_att/spatial_att/dense_3/Tensordot/GatherV2GatherV25conv_att/spatial_att/dense_3/Tensordot/Shape:output:04conv_att/spatial_att/dense_3/Tensordot/free:output:0=conv_att/spatial_att/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6conv_att/spatial_att/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
1conv_att/spatial_att/dense_3/Tensordot/GatherV2_1GatherV25conv_att/spatial_att/dense_3/Tensordot/Shape:output:04conv_att/spatial_att/dense_3/Tensordot/axes:output:0?conv_att/spatial_att/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,conv_att/spatial_att/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
+conv_att/spatial_att/dense_3/Tensordot/ProdProd8conv_att/spatial_att/dense_3/Tensordot/GatherV2:output:05conv_att/spatial_att/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.conv_att/spatial_att/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ë
-conv_att/spatial_att/dense_3/Tensordot/Prod_1Prod:conv_att/spatial_att/dense_3/Tensordot/GatherV2_1:output:07conv_att/spatial_att/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2conv_att/spatial_att/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-conv_att/spatial_att/dense_3/Tensordot/concatConcatV24conv_att/spatial_att/dense_3/Tensordot/free:output:04conv_att/spatial_att/dense_3/Tensordot/axes:output:0;conv_att/spatial_att/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ð
,conv_att/spatial_att/dense_3/Tensordot/stackPack4conv_att/spatial_att/dense_3/Tensordot/Prod:output:06conv_att/spatial_att/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:â
0conv_att/spatial_att/dense_3/Tensordot/transpose	Transpose1conv_att/spatial_att/re_lu_1/Relu_1:activations:06conv_att/spatial_att/dense_3/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdá
.conv_att/spatial_att/dense_3/Tensordot/ReshapeReshape4conv_att/spatial_att/dense_3/Tensordot/transpose:y:05conv_att/spatial_att/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
-conv_att/spatial_att/dense_3/Tensordot/MatMulMatMul7conv_att/spatial_att/dense_3/Tensordot/Reshape:output:0=conv_att/spatial_att/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
.conv_att/spatial_att/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:v
4conv_att/spatial_att/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
/conv_att/spatial_att/dense_3/Tensordot/concat_1ConcatV28conv_att/spatial_att/dense_3/Tensordot/GatherV2:output:07conv_att/spatial_att/dense_3/Tensordot/Const_2:output:0=conv_att/spatial_att/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Þ
&conv_att/spatial_att/dense_3/TensordotReshape7conv_att/spatial_att/dense_3/Tensordot/MatMul:product:08conv_att/spatial_att/dense_3/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd´
5conv_att/spatial_att/dense_2/Tensordot/ReadVariableOpReadVariableOp>conv_att_spatial_att_dense_2_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0u
+conv_att/spatial_att/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+conv_att/spatial_att/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
,conv_att/spatial_att/dense_2/Tensordot/ShapeShape%conv_att/re_lu_3/Relu_1:activations:0*
T0*
_output_shapes
:v
4conv_att/spatial_att/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¯
/conv_att/spatial_att/dense_2/Tensordot/GatherV2GatherV25conv_att/spatial_att/dense_2/Tensordot/Shape:output:04conv_att/spatial_att/dense_2/Tensordot/free:output:0=conv_att/spatial_att/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6conv_att/spatial_att/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
1conv_att/spatial_att/dense_2/Tensordot/GatherV2_1GatherV25conv_att/spatial_att/dense_2/Tensordot/Shape:output:04conv_att/spatial_att/dense_2/Tensordot/axes:output:0?conv_att/spatial_att/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,conv_att/spatial_att/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
+conv_att/spatial_att/dense_2/Tensordot/ProdProd8conv_att/spatial_att/dense_2/Tensordot/GatherV2:output:05conv_att/spatial_att/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.conv_att/spatial_att/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ë
-conv_att/spatial_att/dense_2/Tensordot/Prod_1Prod:conv_att/spatial_att/dense_2/Tensordot/GatherV2_1:output:07conv_att/spatial_att/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2conv_att/spatial_att/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-conv_att/spatial_att/dense_2/Tensordot/concatConcatV24conv_att/spatial_att/dense_2/Tensordot/free:output:04conv_att/spatial_att/dense_2/Tensordot/axes:output:0;conv_att/spatial_att/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ð
,conv_att/spatial_att/dense_2/Tensordot/stackPack4conv_att/spatial_att/dense_2/Tensordot/Prod:output:06conv_att/spatial_att/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ò
0conv_att/spatial_att/dense_2/Tensordot/transpose	Transpose%conv_att/re_lu_3/Relu_1:activations:06conv_att/spatial_att/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 á
.conv_att/spatial_att/dense_2/Tensordot/ReshapeReshape4conv_att/spatial_att/dense_2/Tensordot/transpose:y:05conv_att/spatial_att/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
-conv_att/spatial_att/dense_2/Tensordot/MatMulMatMul7conv_att/spatial_att/dense_2/Tensordot/Reshape:output:0=conv_att/spatial_att/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
.conv_att/spatial_att/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:v
4conv_att/spatial_att/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
/conv_att/spatial_att/dense_2/Tensordot/concat_1ConcatV28conv_att/spatial_att/dense_2/Tensordot/GatherV2:output:07conv_att/spatial_att/dense_2/Tensordot/Const_2:output:0=conv_att/spatial_att/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ú
&conv_att/spatial_att/dense_2/TensordotReshape7conv_att/spatial_att/dense_2/Tensordot/MatMul:product:08conv_att/spatial_att/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
e
#conv_att/spatial_att/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Æ
conv_att/spatial_att/ExpandDims
ExpandDims/conv_att/spatial_att/dense_2/Tensordot:output:0,conv_att/spatial_att/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
conv_att/spatial_att/addAddV2/conv_att/spatial_att/dense_3/Tensordot:output:0(conv_att/spatial_att/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
$conv_att/spatial_att/activation/TanhTanhconv_att/spatial_att/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d´
5conv_att/spatial_att/dense_4/Tensordot/ReadVariableOpReadVariableOp>conv_att_spatial_att_dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0u
+conv_att/spatial_att/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
+conv_att/spatial_att/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          
,conv_att/spatial_att/dense_4/Tensordot/ShapeShape(conv_att/spatial_att/activation/Tanh:y:0*
T0*
_output_shapes
:v
4conv_att/spatial_att/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¯
/conv_att/spatial_att/dense_4/Tensordot/GatherV2GatherV25conv_att/spatial_att/dense_4/Tensordot/Shape:output:04conv_att/spatial_att/dense_4/Tensordot/free:output:0=conv_att/spatial_att/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6conv_att/spatial_att/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
1conv_att/spatial_att/dense_4/Tensordot/GatherV2_1GatherV25conv_att/spatial_att/dense_4/Tensordot/Shape:output:04conv_att/spatial_att/dense_4/Tensordot/axes:output:0?conv_att/spatial_att/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,conv_att/spatial_att/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
+conv_att/spatial_att/dense_4/Tensordot/ProdProd8conv_att/spatial_att/dense_4/Tensordot/GatherV2:output:05conv_att/spatial_att/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.conv_att/spatial_att/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ë
-conv_att/spatial_att/dense_4/Tensordot/Prod_1Prod:conv_att/spatial_att/dense_4/Tensordot/GatherV2_1:output:07conv_att/spatial_att/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2conv_att/spatial_att/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-conv_att/spatial_att/dense_4/Tensordot/concatConcatV24conv_att/spatial_att/dense_4/Tensordot/free:output:04conv_att/spatial_att/dense_4/Tensordot/axes:output:0;conv_att/spatial_att/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ð
,conv_att/spatial_att/dense_4/Tensordot/stackPack4conv_att/spatial_att/dense_4/Tensordot/Prod:output:06conv_att/spatial_att/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ù
0conv_att/spatial_att/dense_4/Tensordot/transpose	Transpose(conv_att/spatial_att/activation/Tanh:y:06conv_att/spatial_att/dense_4/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dá
.conv_att/spatial_att/dense_4/Tensordot/ReshapeReshape4conv_att/spatial_att/dense_4/Tensordot/transpose:y:05conv_att/spatial_att/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
-conv_att/spatial_att/dense_4/Tensordot/MatMulMatMul7conv_att/spatial_att/dense_4/Tensordot/Reshape:output:0=conv_att/spatial_att/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
.conv_att/spatial_att/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:v
4conv_att/spatial_att/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
/conv_att/spatial_att/dense_4/Tensordot/concat_1ConcatV28conv_att/spatial_att/dense_4/Tensordot/GatherV2:output:07conv_att/spatial_att/dense_4/Tensordot/Const_2:output:0=conv_att/spatial_att/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Þ
&conv_att/spatial_att/dense_4/TensordotReshape7conv_att/spatial_att/dense_4/Tensordot/MatMul:product:08conv_att/spatial_att/dense_4/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d®
conv_att/spatial_att/SqueezeSqueeze/conv_att/spatial_att/dense_4/Tensordot:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ
$conv_att/spatial_att/softmax/SoftmaxSoftmax%conv_att/spatial_att/Squeeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d[
conv_att/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :±
conv_att/ExpandDims_3
ExpandDims.conv_att/spatial_att/softmax/Softmax:softmax:0"conv_att/ExpandDims_3/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
conv_att/mulMulconv_att/ExpandDims_3:output:0conv_att/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d~
3conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿË
/conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims
ExpandDimsconv_att/mul:z:0<conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
dÎ
@conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpIconv_att_enc_forward_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0w
5conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims_1
ExpandDimsHconv_att/enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp:value:0>conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
*conv_att/enc_forward/conv1d_3/Conv1D/ShapeShape8conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:
8conv_att/enc_forward/conv1d_3/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:conv_att/enc_forward/conv1d_3/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
:conv_att/enc_forward/conv1d_3/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2conv_att/enc_forward/conv1d_3/Conv1D/strided_sliceStridedSlice3conv_att/enc_forward/conv1d_3/Conv1D/Shape:output:0Aconv_att/enc_forward/conv1d_3/Conv1D/strided_slice/stack:output:0Cconv_att/enc_forward/conv1d_3/Conv1D/strided_slice/stack_1:output:0Cconv_att/enc_forward/conv1d_3/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
2conv_att/enc_forward/conv1d_3/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   d      è
,conv_att/enc_forward/conv1d_3/Conv1D/ReshapeReshape8conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims:output:0;conv_att/enc_forward/conv1d_3/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
+conv_att/enc_forward/conv1d_3/Conv1D/Conv2DConv2D5conv_att/enc_forward/conv1d_3/Conv1D/Reshape:output:0:conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb *
paddingVALID*
strides

4conv_att/enc_forward/conv1d_3/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   b       {
0conv_att/enc_forward/conv1d_3/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
+conv_att/enc_forward/conv1d_3/Conv1D/concatConcatV2;conv_att/enc_forward/conv1d_3/Conv1D/strided_slice:output:0=conv_att/enc_forward/conv1d_3/Conv1D/concat/values_1:output:09conv_att/enc_forward/conv1d_3/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:ã
.conv_att/enc_forward/conv1d_3/Conv1D/Reshape_1Reshape4conv_att/enc_forward/conv1d_3/Conv1D/Conv2D:output:04conv_att/enc_forward/conv1d_3/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b Ê
,conv_att/enc_forward/conv1d_3/Conv1D/SqueezeSqueeze7conv_att/enc_forward/conv1d_3/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
6conv_att/enc_forward/conv1d_3/squeeze_batch_dims/ShapeShape5conv_att/enc_forward/conv1d_3/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:
Dconv_att/enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fconv_att/enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
Fconv_att/enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
>conv_att/enc_forward/conv1d_3/squeeze_batch_dims/strided_sliceStridedSlice?conv_att/enc_forward/conv1d_3/squeeze_batch_dims/Shape:output:0Mconv_att/enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack:output:0Oconv_att/enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_1:output:0Oconv_att/enc_forward/conv1d_3/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
>conv_att/enc_forward/conv1d_3/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿb       ù
8conv_att/enc_forward/conv1d_3/squeeze_batch_dims/ReshapeReshape5conv_att/enc_forward/conv1d_3/Conv1D/Squeeze:output:0Gconv_att/enc_forward/conv1d_3/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb Ô
Gconv_att/enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpPconv_att_enc_forward_conv1d_3_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
8conv_att/enc_forward/conv1d_3/squeeze_batch_dims/BiasAddBiasAddAconv_att/enc_forward/conv1d_3/squeeze_batch_dims/Reshape:output:0Oconv_att/enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿb 
@conv_att/enc_forward/conv1d_3/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"b       
<conv_att/enc_forward/conv1d_3/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
7conv_att/enc_forward/conv1d_3/squeeze_batch_dims/concatConcatV2Gconv_att/enc_forward/conv1d_3/squeeze_batch_dims/strided_slice:output:0Iconv_att/enc_forward/conv1d_3/squeeze_batch_dims/concat/values_1:output:0Econv_att/enc_forward/conv1d_3/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:
:conv_att/enc_forward/conv1d_3/squeeze_batch_dims/Reshape_1ReshapeAconv_att/enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd:output:0@conv_att/enc_forward/conv1d_3/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b ¨
!conv_att/enc_forward/re_lu_2/ReluReluCconv_att/enc_forward/conv1d_3/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b ~
3conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿê
/conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims
ExpandDims/conv_att/enc_forward/re_lu_2/Relu:activations:0<conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b Î
@conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpIconv_att_enc_forward_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0w
5conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims_1
ExpandDimsHconv_att/enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0>conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  
*conv_att/enc_forward/conv1d_4/Conv1D/ShapeShape8conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:
8conv_att/enc_forward/conv1d_4/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:conv_att/enc_forward/conv1d_4/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
:conv_att/enc_forward/conv1d_4/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2conv_att/enc_forward/conv1d_4/Conv1D/strided_sliceStridedSlice3conv_att/enc_forward/conv1d_4/Conv1D/Shape:output:0Aconv_att/enc_forward/conv1d_4/Conv1D/strided_slice/stack:output:0Cconv_att/enc_forward/conv1d_4/Conv1D/strided_slice/stack_1:output:0Cconv_att/enc_forward/conv1d_4/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
2conv_att/enc_forward/conv1d_4/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   b       è
,conv_att/enc_forward/conv1d_4/Conv1D/ReshapeReshape8conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims:output:0;conv_att/enc_forward/conv1d_4/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb 
+conv_att/enc_forward/conv1d_4/Conv1D/Conv2DConv2D5conv_att/enc_forward/conv1d_4/Conv1D/Reshape:output:0:conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` *
paddingVALID*
strides

4conv_att/enc_forward/conv1d_4/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   `       {
0conv_att/enc_forward/conv1d_4/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
+conv_att/enc_forward/conv1d_4/Conv1D/concatConcatV2;conv_att/enc_forward/conv1d_4/Conv1D/strided_slice:output:0=conv_att/enc_forward/conv1d_4/Conv1D/concat/values_1:output:09conv_att/enc_forward/conv1d_4/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:ã
.conv_att/enc_forward/conv1d_4/Conv1D/Reshape_1Reshape4conv_att/enc_forward/conv1d_4/Conv1D/Conv2D:output:04conv_att/enc_forward/conv1d_4/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` Ê
,conv_att/enc_forward/conv1d_4/Conv1D/SqueezeSqueeze7conv_att/enc_forward/conv1d_4/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
6conv_att/enc_forward/conv1d_4/squeeze_batch_dims/ShapeShape5conv_att/enc_forward/conv1d_4/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:
Dconv_att/enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fconv_att/enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
Fconv_att/enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
>conv_att/enc_forward/conv1d_4/squeeze_batch_dims/strided_sliceStridedSlice?conv_att/enc_forward/conv1d_4/squeeze_batch_dims/Shape:output:0Mconv_att/enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack:output:0Oconv_att/enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_1:output:0Oconv_att/enc_forward/conv1d_4/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
>conv_att/enc_forward/conv1d_4/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ`       ù
8conv_att/enc_forward/conv1d_4/squeeze_batch_dims/ReshapeReshape5conv_att/enc_forward/conv1d_4/Conv1D/Squeeze:output:0Gconv_att/enc_forward/conv1d_4/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` Ô
Gconv_att/enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpPconv_att_enc_forward_conv1d_4_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
8conv_att/enc_forward/conv1d_4/squeeze_batch_dims/BiasAddBiasAddAconv_att/enc_forward/conv1d_4/squeeze_batch_dims/Reshape:output:0Oconv_att/enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` 
@conv_att/enc_forward/conv1d_4/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"`       
<conv_att/enc_forward/conv1d_4/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
7conv_att/enc_forward/conv1d_4/squeeze_batch_dims/concatConcatV2Gconv_att/enc_forward/conv1d_4/squeeze_batch_dims/strided_slice:output:0Iconv_att/enc_forward/conv1d_4/squeeze_batch_dims/concat/values_1:output:0Econv_att/enc_forward/conv1d_4/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:
:conv_att/enc_forward/conv1d_4/squeeze_batch_dims/Reshape_1ReshapeAconv_att/enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd:output:0@conv_att/enc_forward/conv1d_4/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` ª
#conv_att/enc_forward/re_lu_2/Relu_1ReluCconv_att/enc_forward/conv1d_4/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` ~
3conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿì
/conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims
ExpandDims1conv_att/enc_forward/re_lu_2/Relu_1:activations:0<conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` Î
@conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpIconv_att_enc_forward_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0w
5conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ú
1conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims_1
ExpandDimsHconv_att/enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0>conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  
*conv_att/enc_forward/conv1d_5/Conv1D/ShapeShape8conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims:output:0*
T0*
_output_shapes
:
8conv_att/enc_forward/conv1d_5/Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:conv_att/enc_forward/conv1d_5/Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
:conv_att/enc_forward/conv1d_5/Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2conv_att/enc_forward/conv1d_5/Conv1D/strided_sliceStridedSlice3conv_att/enc_forward/conv1d_5/Conv1D/Shape:output:0Aconv_att/enc_forward/conv1d_5/Conv1D/strided_slice/stack:output:0Cconv_att/enc_forward/conv1d_5/Conv1D/strided_slice/stack_1:output:0Cconv_att/enc_forward/conv1d_5/Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
2conv_att/enc_forward/conv1d_5/Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   `       è
,conv_att/enc_forward/conv1d_5/Conv1D/ReshapeReshape8conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims:output:0;conv_att/enc_forward/conv1d_5/Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` 
+conv_att/enc_forward/conv1d_5/Conv1D/Conv2DConv2D5conv_att/enc_forward/conv1d_5/Conv1D/Reshape:output:0:conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ *
paddingVALID*
strides

4conv_att/enc_forward/conv1d_5/Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   ^       {
0conv_att/enc_forward/conv1d_5/Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
+conv_att/enc_forward/conv1d_5/Conv1D/concatConcatV2;conv_att/enc_forward/conv1d_5/Conv1D/strided_slice:output:0=conv_att/enc_forward/conv1d_5/Conv1D/concat/values_1:output:09conv_att/enc_forward/conv1d_5/Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:ã
.conv_att/enc_forward/conv1d_5/Conv1D/Reshape_1Reshape4conv_att/enc_forward/conv1d_5/Conv1D/Conv2D:output:04conv_att/enc_forward/conv1d_5/Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
^ Ê
,conv_att/enc_forward/conv1d_5/Conv1D/SqueezeSqueeze7conv_att/enc_forward/conv1d_5/Conv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
6conv_att/enc_forward/conv1d_5/squeeze_batch_dims/ShapeShape5conv_att/enc_forward/conv1d_5/Conv1D/Squeeze:output:0*
T0*
_output_shapes
:
Dconv_att/enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fconv_att/enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
Fconv_att/enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
>conv_att/enc_forward/conv1d_5/squeeze_batch_dims/strided_sliceStridedSlice?conv_att/enc_forward/conv1d_5/squeeze_batch_dims/Shape:output:0Mconv_att/enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack:output:0Oconv_att/enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_1:output:0Oconv_att/enc_forward/conv1d_5/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
>conv_att/enc_forward/conv1d_5/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ^       ù
8conv_att/enc_forward/conv1d_5/squeeze_batch_dims/ReshapeReshape5conv_att/enc_forward/conv1d_5/Conv1D/Squeeze:output:0Gconv_att/enc_forward/conv1d_5/squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ Ô
Gconv_att/enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpPconv_att_enc_forward_conv1d_5_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
8conv_att/enc_forward/conv1d_5/squeeze_batch_dims/BiasAddBiasAddAconv_att/enc_forward/conv1d_5/squeeze_batch_dims/Reshape:output:0Oconv_att/enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ 
@conv_att/enc_forward/conv1d_5/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"^       
<conv_att/enc_forward/conv1d_5/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
7conv_att/enc_forward/conv1d_5/squeeze_batch_dims/concatConcatV2Gconv_att/enc_forward/conv1d_5/squeeze_batch_dims/strided_slice:output:0Iconv_att/enc_forward/conv1d_5/squeeze_batch_dims/concat/values_1:output:0Econv_att/enc_forward/conv1d_5/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:
:conv_att/enc_forward/conv1d_5/squeeze_batch_dims/Reshape_1ReshapeAconv_att/enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd:output:0@conv_att/enc_forward/conv1d_5/squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ ª
#conv_att/enc_forward/re_lu_2/Relu_2ReluCconv_att/enc_forward/conv1d_5/squeeze_batch_dims/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ k
conv_att/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ
   À  ¦
conv_att/ReshapeReshape1conv_att/enc_forward/re_lu_2/Relu_2:activations:0conv_att/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
)conv_att/dense_7/Tensordot/ReadVariableOpReadVariableOp2conv_att_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	À *
dtype0i
conv_att/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
conv_att/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       i
 conv_att/dense_7/Tensordot/ShapeShapeconv_att/Reshape:output:0*
T0*
_output_shapes
:j
(conv_att/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#conv_att/dense_7/Tensordot/GatherV2GatherV2)conv_att/dense_7/Tensordot/Shape:output:0(conv_att/dense_7/Tensordot/free:output:01conv_att/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*conv_att/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%conv_att/dense_7/Tensordot/GatherV2_1GatherV2)conv_att/dense_7/Tensordot/Shape:output:0(conv_att/dense_7/Tensordot/axes:output:03conv_att/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 conv_att/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
conv_att/dense_7/Tensordot/ProdProd,conv_att/dense_7/Tensordot/GatherV2:output:0)conv_att/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"conv_att/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!conv_att/dense_7/Tensordot/Prod_1Prod.conv_att/dense_7/Tensordot/GatherV2_1:output:0+conv_att/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&conv_att/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!conv_att/dense_7/Tensordot/concatConcatV2(conv_att/dense_7/Tensordot/free:output:0(conv_att/dense_7/Tensordot/axes:output:0/conv_att/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 conv_att/dense_7/Tensordot/stackPack(conv_att/dense_7/Tensordot/Prod:output:0*conv_att/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¯
$conv_att/dense_7/Tensordot/transpose	Transposeconv_att/Reshape:output:0*conv_att/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À½
"conv_att/dense_7/Tensordot/ReshapeReshape(conv_att/dense_7/Tensordot/transpose:y:0)conv_att/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
!conv_att/dense_7/Tensordot/MatMulMatMul+conv_att/dense_7/Tensordot/Reshape:output:01conv_att/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
"conv_att/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: j
(conv_att/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#conv_att/dense_7/Tensordot/concat_1ConcatV2,conv_att/dense_7/Tensordot/GatherV2:output:0+conv_att/dense_7/Tensordot/Const_2:output:01conv_att/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¶
conv_att/dense_7/TensordotReshape+conv_att/dense_7/Tensordot/MatMul:product:0,conv_att/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
'conv_att/dense_7/BiasAdd/ReadVariableOpReadVariableOp0conv_att_dense_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¯
conv_att/dense_7/BiasAddBiasAdd#conv_att/dense_7/Tensordot:output:0/conv_att/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 x
conv_att/re_lu_3/Relu_2Relu!conv_att/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 a
conv_att/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿË
conv_att/concat_2ConcatV2%conv_att/re_lu_3/Relu_1:activations:0%conv_att/re_lu_3/Relu_2:activations:0conv_att/concat_2/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@
)conv_att/dense_8/Tensordot/ReadVariableOpReadVariableOp2conv_att_dense_8_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0i
conv_att/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
conv_att/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       j
 conv_att/dense_8/Tensordot/ShapeShapeconv_att/concat_2:output:0*
T0*
_output_shapes
:j
(conv_att/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#conv_att/dense_8/Tensordot/GatherV2GatherV2)conv_att/dense_8/Tensordot/Shape:output:0(conv_att/dense_8/Tensordot/free:output:01conv_att/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*conv_att/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%conv_att/dense_8/Tensordot/GatherV2_1GatherV2)conv_att/dense_8/Tensordot/Shape:output:0(conv_att/dense_8/Tensordot/axes:output:03conv_att/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 conv_att/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
conv_att/dense_8/Tensordot/ProdProd,conv_att/dense_8/Tensordot/GatherV2:output:0)conv_att/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"conv_att/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!conv_att/dense_8/Tensordot/Prod_1Prod.conv_att/dense_8/Tensordot/GatherV2_1:output:0+conv_att/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&conv_att/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!conv_att/dense_8/Tensordot/concatConcatV2(conv_att/dense_8/Tensordot/free:output:0(conv_att/dense_8/Tensordot/axes:output:0/conv_att/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 conv_att/dense_8/Tensordot/stackPack(conv_att/dense_8/Tensordot/Prod:output:0*conv_att/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¯
$conv_att/dense_8/Tensordot/transpose	Transposeconv_att/concat_2:output:0*conv_att/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@½
"conv_att/dense_8/Tensordot/ReshapeReshape(conv_att/dense_8/Tensordot/transpose:y:0)conv_att/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
!conv_att/dense_8/Tensordot/MatMulMatMul+conv_att/dense_8/Tensordot/Reshape:output:01conv_att/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
"conv_att/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: j
(conv_att/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#conv_att/dense_8/Tensordot/concat_1ConcatV2,conv_att/dense_8/Tensordot/GatherV2:output:0+conv_att/dense_8/Tensordot/Const_2:output:01conv_att/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¶
conv_att/dense_8/TensordotReshape+conv_att/dense_8/Tensordot/MatMul:product:0,conv_att/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
'conv_att/dense_8/BiasAdd/ReadVariableOpReadVariableOp0conv_att_dense_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¯
conv_att/dense_8/BiasAddBiasAdd#conv_att/dense_8/Tensordot:output:0/conv_att/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 x
conv_att/re_lu_3/Relu_3Relu!conv_att/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
)conv_att/dense_9/Tensordot/ReadVariableOpReadVariableOp2conv_att_dense_9_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0i
conv_att/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
conv_att/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
 conv_att/dense_9/Tensordot/ShapeShape%conv_att/re_lu_3/Relu_3:activations:0*
T0*
_output_shapes
:j
(conv_att/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
#conv_att/dense_9/Tensordot/GatherV2GatherV2)conv_att/dense_9/Tensordot/Shape:output:0(conv_att/dense_9/Tensordot/free:output:01conv_att/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*conv_att/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%conv_att/dense_9/Tensordot/GatherV2_1GatherV2)conv_att/dense_9/Tensordot/Shape:output:0(conv_att/dense_9/Tensordot/axes:output:03conv_att/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 conv_att/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
conv_att/dense_9/Tensordot/ProdProd,conv_att/dense_9/Tensordot/GatherV2:output:0)conv_att/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"conv_att/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: §
!conv_att/dense_9/Tensordot/Prod_1Prod.conv_att/dense_9/Tensordot/GatherV2_1:output:0+conv_att/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&conv_att/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : à
!conv_att/dense_9/Tensordot/concatConcatV2(conv_att/dense_9/Tensordot/free:output:0(conv_att/dense_9/Tensordot/axes:output:0/conv_att/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¬
 conv_att/dense_9/Tensordot/stackPack(conv_att/dense_9/Tensordot/Prod:output:0*conv_att/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:º
$conv_att/dense_9/Tensordot/transpose	Transpose%conv_att/re_lu_3/Relu_3:activations:0*conv_att/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ½
"conv_att/dense_9/Tensordot/ReshapeReshape(conv_att/dense_9/Tensordot/transpose:y:0)conv_att/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
!conv_att/dense_9/Tensordot/MatMulMatMul+conv_att/dense_9/Tensordot/Reshape:output:01conv_att/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"conv_att/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(conv_att/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
#conv_att/dense_9/Tensordot/concat_1ConcatV2,conv_att/dense_9/Tensordot/GatherV2:output:0+conv_att/dense_9/Tensordot/Const_2:output:01conv_att/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¶
conv_att/dense_9/TensordotReshape+conv_att/dense_9/Tensordot/MatMul:product:0,conv_att/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'conv_att/dense_9/BiasAdd/ReadVariableOpReadVariableOp0conv_att_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¯
conv_att/dense_9/BiasAddBiasAdd#conv_att/dense_9/Tensordot:output:0/conv_att/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
{
conv_att/activation_1/TanhTanh!conv_att/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
conv_att/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿÿÿÿÿs
conv_att/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
conv_att/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
conv_att/strided_sliceStridedSliceargs_0%conv_att/strided_slice/stack:output:0'conv_att/strided_slice/stack_1:output:0'conv_att/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
conv_att/addAddV2conv_att/activation_1/Tanh:y:0conv_att/strided_slice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c
IdentityIdentityconv_att/add:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ð
NoOpNoOp(^conv_att/dense_5/BiasAdd/ReadVariableOp'^conv_att/dense_5/MatMul/ReadVariableOp(^conv_att/dense_6/BiasAdd/ReadVariableOp*^conv_att/dense_6/Tensordot/ReadVariableOp(^conv_att/dense_7/BiasAdd/ReadVariableOp*^conv_att/dense_7/Tensordot/ReadVariableOp(^conv_att/dense_8/BiasAdd/ReadVariableOp*^conv_att/dense_8/Tensordot/ReadVariableOp(^conv_att/dense_9/BiasAdd/ReadVariableOp*^conv_att/dense_9/Tensordot/ReadVariableOpA^conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOpH^conv_att/enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpA^conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpH^conv_att/enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpA^conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpH^conv_att/enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp0^conv_att/enc_past/conv1d/BiasAdd/ReadVariableOp<^conv_att/enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2^conv_att/enc_past/conv1d_1/BiasAdd/ReadVariableOp>^conv_att/enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2^conv_att/enc_past/conv1d_2/BiasAdd/ReadVariableOp>^conv_att/enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2^conv_att/spatial_att/dense/BiasAdd/ReadVariableOp4^conv_att/spatial_att/dense/Tensordot/ReadVariableOp4^conv_att/spatial_att/dense_1/BiasAdd/ReadVariableOp6^conv_att/spatial_att/dense_1/Tensordot/ReadVariableOp6^conv_att/spatial_att/dense_2/Tensordot/ReadVariableOp6^conv_att/spatial_att/dense_3/Tensordot/ReadVariableOp6^conv_att/spatial_att/dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*«
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'conv_att/dense_5/BiasAdd/ReadVariableOp'conv_att/dense_5/BiasAdd/ReadVariableOp2P
&conv_att/dense_5/MatMul/ReadVariableOp&conv_att/dense_5/MatMul/ReadVariableOp2R
'conv_att/dense_6/BiasAdd/ReadVariableOp'conv_att/dense_6/BiasAdd/ReadVariableOp2V
)conv_att/dense_6/Tensordot/ReadVariableOp)conv_att/dense_6/Tensordot/ReadVariableOp2R
'conv_att/dense_7/BiasAdd/ReadVariableOp'conv_att/dense_7/BiasAdd/ReadVariableOp2V
)conv_att/dense_7/Tensordot/ReadVariableOp)conv_att/dense_7/Tensordot/ReadVariableOp2R
'conv_att/dense_8/BiasAdd/ReadVariableOp'conv_att/dense_8/BiasAdd/ReadVariableOp2V
)conv_att/dense_8/Tensordot/ReadVariableOp)conv_att/dense_8/Tensordot/ReadVariableOp2R
'conv_att/dense_9/BiasAdd/ReadVariableOp'conv_att/dense_9/BiasAdd/ReadVariableOp2V
)conv_att/dense_9/Tensordot/ReadVariableOp)conv_att/dense_9/Tensordot/ReadVariableOp2
@conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp@conv_att/enc_forward/conv1d_3/Conv1D/ExpandDims_1/ReadVariableOp2
Gconv_att/enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOpGconv_att/enc_forward/conv1d_3/squeeze_batch_dims/BiasAdd/ReadVariableOp2
@conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp@conv_att/enc_forward/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2
Gconv_att/enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOpGconv_att/enc_forward/conv1d_4/squeeze_batch_dims/BiasAdd/ReadVariableOp2
@conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp@conv_att/enc_forward/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2
Gconv_att/enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOpGconv_att/enc_forward/conv1d_5/squeeze_batch_dims/BiasAdd/ReadVariableOp2b
/conv_att/enc_past/conv1d/BiasAdd/ReadVariableOp/conv_att/enc_past/conv1d/BiasAdd/ReadVariableOp2z
;conv_att/enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp;conv_att/enc_past/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2f
1conv_att/enc_past/conv1d_1/BiasAdd/ReadVariableOp1conv_att/enc_past/conv1d_1/BiasAdd/ReadVariableOp2~
=conv_att/enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp=conv_att/enc_past/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2f
1conv_att/enc_past/conv1d_2/BiasAdd/ReadVariableOp1conv_att/enc_past/conv1d_2/BiasAdd/ReadVariableOp2~
=conv_att/enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp=conv_att/enc_past/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2f
1conv_att/spatial_att/dense/BiasAdd/ReadVariableOp1conv_att/spatial_att/dense/BiasAdd/ReadVariableOp2j
3conv_att/spatial_att/dense/Tensordot/ReadVariableOp3conv_att/spatial_att/dense/Tensordot/ReadVariableOp2j
3conv_att/spatial_att/dense_1/BiasAdd/ReadVariableOp3conv_att/spatial_att/dense_1/BiasAdd/ReadVariableOp2n
5conv_att/spatial_att/dense_1/Tensordot/ReadVariableOp5conv_att/spatial_att/dense_1/Tensordot/ReadVariableOp2n
5conv_att/spatial_att/dense_2/Tensordot/ReadVariableOp5conv_att/spatial_att/dense_2/Tensordot/ReadVariableOp2n
5conv_att/spatial_att/dense_3/Tensordot/ReadVariableOp5conv_att/spatial_att/dense_3/Tensordot/ReadVariableOp2n
5conv_att/spatial_att/dense_4/Tensordot/ReadVariableOp5conv_att/spatial_att/dense_4/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameargs_1:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_2:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameargs_3

£
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629067
x&
conv1d_3_1628956: 
conv1d_3_1628958: &
conv1d_4_1629006:  
conv1d_4_1629008: &
conv1d_5_1629055:  
conv1d_5_1629057: 
identity¢ conv1d_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCallÿ
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCallxconv1d_3_1628956conv1d_3_1628958*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1628955ë
re_lu_2/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1628966
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:0conv1d_4_1629006conv1d_4_1629008*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1629005í
re_lu_2/PartitionedCall_1PartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629015 
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall"re_lu_2/PartitionedCall_1:output:0conv1d_5_1629055conv1d_5_1629057*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1629054í
re_lu_2/PartitionedCall_2PartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629064y
IdentityIdentity"re_lu_2/PartitionedCall_2:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ ¯
NoOpNoOp!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ
d: : : : : : 2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d

_user_specified_namex
¿0
Ó
E__inference_enc_past_layer_call_and_return_conditional_losses_1631271
xH
2conv1d_conv1d_expanddims_1_readvariableop_resource: 4
&conv1d_biasadd_readvariableop_resource: J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:  6
(conv1d_1_biasadd_readvariableop_resource: J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:  6
(conv1d_2_biasadd_readvariableop_resource: 
identity¢conv1d/BiasAdd/ReadVariableOp¢)conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_2/BiasAdd/ReadVariableOp¢+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d/Conv1D/ExpandDims
ExpandDimsx%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : µ
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Â
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a

re_lu/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¥
conv1d_1/Conv1D/ExpandDims
ExpandDimsre_lu/Relu:activations:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  È
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
re_lu/Relu_1Reluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ§
conv1d_2/Conv1D/ExpandDims
ExpandDimsre_lu/Relu_1:activations:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  È
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
re_lu/Relu_2Reluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
IdentityIdentityre_lu/Relu_2:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ²
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ó
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1632030

inputs
identityJ
TanhTanhinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
T
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
á

*__inference_conv1d_1_layer_call_fn_1632134

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1628693s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
è
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1628966

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
b :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b 
 
_user_specified_nameinputs
²
C
'__inference_re_lu_layer_call_fn_1632051

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628703d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
è
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1632096

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
 
_user_specified_nameinputs
Ö
^
B__inference_re_lu_layer_call_and_return_conditional_losses_1632066

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Æ
E
)__inference_re_lu_2_layer_call_fn_1632081

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629015h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
 
_user_specified_nameinputs
	

*__inference_enc_past_layer_call_fn_1628748
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_enc_past_layer_call_and_return_conditional_losses_1628733s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ª	

-__inference_enc_forward_layer_call_fn_1629082
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *Q
fLRJ
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629067w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ
d: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
!
_user_specified_name	input_1
í
Á
%__inference_signature_wrapper_1631157

args_0

args_1

args_2

args_3
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:	À 
	unknown_6: 
	unknown_7:$ 
	unknown_8: 
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17:  

unknown_18:  

unknown_19:  

unknown_20:  

unknown_21: 

unknown_22:	À 

unknown_23: 

unknown_24:@ 

unknown_25: 

unknown_26: 

unknown_27:
identity¢StatefulPartitionedCallÐ
StatefulPartitionedCallStatefulPartitionedCallargs_0args_1args_2args_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*?
_read_only_resource_inputs!
	
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *+
f&R$
"__inference__wrapped_model_1628643s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*«
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameargs_1:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_2:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameargs_3
Æ
E
)__inference_re_lu_2_layer_call_fn_1632086

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *M
fHRF
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1628966h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
b :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b 
 
_user_specified_nameinputs
Ý)
º
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1632311

inputsA
+conv1d_expanddims_1_readvariableop_resource:  @
2squeeze_batch_dims_biasadd_readvariableop_resource: 
identity¢"Conv1D/ExpandDims_1/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  V
Conv1D/ShapeShapeConv1D/ExpandDims:output:0*
T0*
_output_shapes
:d
Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿf
Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Conv1D/strided_sliceStridedSliceConv1D/Shape:output:0#Conv1D/strided_slice/stack:output:0%Conv1D/strided_slice/stack_1:output:0%Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   `       
Conv1D/ReshapeReshapeConv1D/ExpandDims:output:0Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` ±
Conv1D/Conv2DConv2DConv1D/Reshape:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ *
paddingVALID*
strides
k
Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   ^       ]
Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
Conv1D/concatConcatV2Conv1D/strided_slice:output:0Conv1D/concat/values_1:output:0Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:
Conv1D/Reshape_1ReshapeConv1D/Conv2D:output:0Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
^ 
Conv1D/SqueezeSqueezeConv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ_
squeeze_batch_dims/ShapeShapeConv1D/Squeeze:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿr
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ^       
squeeze_batch_dims/ReshapeReshapeConv1D/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ 
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^ s
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"^       i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÔ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ª
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ |
IdentityIdentity%squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ 
NoOpNoOp#^Conv1D/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
` : : 2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
 
_user_specified_nameinputs
Õ
 
E__inference_enc_past_layer_call_and_return_conditional_losses_1628911
input_1$
conv1d_1628892: 
conv1d_1628894: &
conv1d_1_1628898:  
conv1d_1_1628900: &
conv1d_2_1628904:  
conv1d_2_1628906: 
identity¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCallù
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1628892conv1d_1628894*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_1628665á
re_lu/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628676
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv1d_1_1628898conv1d_1_1628900*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1628693å
re_lu/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628703
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_1:output:0conv1d_2_1628904conv1d_2_1628906*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1628720å
re_lu/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628730s
IdentityIdentity re_lu/PartitionedCall_2:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ö
^
B__inference_re_lu_layer_call_and_return_conditional_losses_1628730

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ð
ü
D__inference_dense_7_layer_call_and_return_conditional_losses_1629554

inputs4
!tensordot_readvariableop_resource:	À -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	À *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
À: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
 
_user_specified_nameinputs
Ö
^
B__inference_re_lu_layer_call_and_return_conditional_losses_1628703

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1629641

inputs
identityJ
TanhTanhinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
T
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¿
Ô
*__inference_conv_att_layer_call_fn_1630193

x_past
	x_forward
pos_enc_past
pos_enc_fwd
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:	À 
	unknown_6: 
	unknown_7:$ 
	unknown_8: 
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17:  

unknown_18:  

unknown_19:  

unknown_20:  

unknown_21: 

unknown_22:	À 

unknown_23: 

unknown_24:@ 

unknown_25: 

unknown_26: 

unknown_27:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallx_past	x_forwardpos_enc_pastpos_enc_fwdunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*?
_read_only_resource_inputs!
	
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv_att_layer_call_and_return_conditional_losses_1629649s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*«
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namex_past:VR
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#
_user_specified_name	x_forward:YU
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

&
_user_specified_namepos_enc_past:XT
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
%
_user_specified_namepos_enc_fwd
¾
`
D__inference_flatten_layer_call_and_return_conditional_losses_1632041

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÀ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
è
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1629064

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
^ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^ 
 
_user_specified_nameinputs

û
H__inference_spatial_att_layer_call_and_return_conditional_losses_1629490	
query
feature9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:;
)dense_3_tensordot_readvariableop_resource:;
)dense_2_tensordot_readvariableop_resource: ;
)dense_4_tensordot_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp¢ dense_3/Tensordot/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          L
dense/Tensordot/ShapeShapefeature*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposefeaturedense/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
re_lu_1/ReluReludense/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          a
dense_1/Tensordot/ShapeShapere_lu_1/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¡
dense_1/Tensordot/transpose	Transposere_lu_1/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdj
re_lu_1/Relu_1Reludense_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          c
dense_3/Tensordot/ShapeShapere_lu_1/Relu_1:activations:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:£
dense_3/Tensordot/transpose	Transposere_lu_1/Relu_1:activations:0!dense_3/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       L
dense_2/Tensordot/ShapeShapequery*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/transpose	Transposequery!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ¢
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsdense_2/Tensordot:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
addAddV2dense_3/Tensordot:output:0ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dZ
activation/TanhTanhadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          Z
dense_4/Tensordot/ShapeShapeactivation/Tanh:y:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposeactivation/Tanh:y:0!dense_4/Tensordot/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d¢
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
SqueezeSqueezedense_4/Tensordot:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d*
squeeze_dims

ÿÿÿÿÿÿÿÿÿb
softmax/SoftmaxSoftmaxSqueeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dl
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d³
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
 :ÿÿÿÿÿÿÿÿÿd: : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp:R N
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 

_user_specified_namequery:XT
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
!
_user_specified_name	feature
Õ
 
E__inference_enc_past_layer_call_and_return_conditional_losses_1628889
input_1$
conv1d_1628870: 
conv1d_1628872: &
conv1d_1_1628876:  
conv1d_1_1628878: &
conv1d_2_1628882:  
conv1d_2_1628884: 
identity¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢ conv1d_2/StatefulPartitionedCallù
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1628870conv1d_1628872*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_1628665á
re_lu/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628676
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv1d_1_1628876conv1d_1_1628878*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1628693å
re_lu/PartitionedCall_1PartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628703
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu/PartitionedCall_1:output:0conv1d_2_1628882conv1d_2_1628884*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1628720å
re_lu/PartitionedCall_2PartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628730s
IdentityIdentity re_lu/PartitionedCall_2:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
NoOpNoOp^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ë	
ö
D__inference_dense_5_layer_call_and_return_conditional_losses_1631844

inputs1
matmul_readvariableop_resource:	À -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	À *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
 
_user_specified_nameinputs
¿0
Ó
E__inference_enc_past_layer_call_and_return_conditional_losses_1631231
xH
2conv1d_conv1d_expanddims_1_readvariableop_resource: 4
&conv1d_biasadd_readvariableop_resource: J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:  6
(conv1d_1_biasadd_readvariableop_resource: J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:  6
(conv1d_2_biasadd_readvariableop_resource: 
identity¢conv1d/BiasAdd/ReadVariableOp¢)conv1d/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_2/BiasAdd/ReadVariableOp¢+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpg
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d/Conv1D/ExpandDims
ExpandDimsx%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : µ
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Â
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a

re_lu/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¥
conv1d_1/Conv1D/ExpandDims
ExpandDimsre_lu/Relu:activations:0'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  È
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
re_lu/Relu_1Reluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ§
conv1d_2/Conv1D/ExpandDims
ExpandDimsre_lu/Relu_1:activations:0'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  È
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
re_lu/Relu_2Reluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
IdentityIdentityre_lu/Relu_2:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ²
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ý)
º
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1629005

inputsA
+conv1d_expanddims_1_readvariableop_resource:  @
2squeeze_batch_dims_biasadd_readvariableop_resource: 
identity¢"Conv1D/ExpandDims_1/ReadVariableOp¢)squeeze_batch_dims/BiasAdd/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
b 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  V
Conv1D/ShapeShapeConv1D/ExpandDims:output:0*
T0*
_output_shapes
:d
Conv1D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
Conv1D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿf
Conv1D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Conv1D/strided_sliceStridedSliceConv1D/Shape:output:0#Conv1D/strided_slice/stack:output:0%Conv1D/strided_slice/stack_1:output:0%Conv1D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskm
Conv1D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ   b       
Conv1D/ReshapeReshapeConv1D/ExpandDims:output:0Conv1D/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿb ±
Conv1D/Conv2DConv2DConv1D/Reshape:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` *
paddingVALID*
strides
k
Conv1D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"   `       ]
Conv1D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¤
Conv1D/concatConcatV2Conv1D/strided_slice:output:0Conv1D/concat/values_1:output:0Conv1D/concat/axis:output:0*
N*
T0*
_output_shapes
:
Conv1D/Reshape_1ReshapeConv1D/Conv2D:output:0Conv1D/concat:output:0*
T0*3
_output_shapes!
:ÿÿÿÿÿÿÿÿÿ
` 
Conv1D/SqueezeSqueezeConv1D/Reshape_1:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` *
squeeze_dims

ýÿÿÿÿÿÿÿÿ_
squeeze_batch_dims/ShapeShapeConv1D/Squeeze:output:0*
T0*
_output_shapes
:p
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿr
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"ÿÿÿÿ`       
squeeze_batch_dims/ReshapeReshapeConv1D/Squeeze:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` 
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ` s
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB"`       i
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÔ
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:ª
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` |
IdentityIdentity%squeeze_batch_dims/Reshape_1:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
` 
NoOpNoOp#^Conv1D/ExpandDims_1/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
b : : 2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b 
 
_user_specified_nameinputs
Ö
^
B__inference_re_lu_layer_call_and_return_conditional_losses_1628676

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	

*__inference_enc_past_layer_call_fn_1631191
x
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *N
fIRG
E__inference_enc_past_layer_call_and_return_conditional_losses_1628835s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
²
C
'__inference_re_lu_layer_call_fn_1632056

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8 *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_1628676d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
è
`
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1632091

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
b :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b 
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*î
serving_defaultÚ
=
args_03
serving_default_args_0:0ÿÿÿÿÿÿÿÿÿ
=
args_13
serving_default_args_1:0ÿÿÿÿÿÿÿÿÿd
=
args_23
serving_default_args_2:0ÿÿÿÿÿÿÿÿÿ

=
args_33
serving_default_args_3:0ÿÿÿÿÿÿÿÿÿd@
output_14
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:ù
ß
enc_past
spatial_att
enc_forward

dense1

dense2

dense3

dense4

dense5
	relu

tanh
flatten
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
ß
filters
	sizes
strides
	convs
relu
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_model
ù

dense1

 dense2
!w_e
"u_e
#v_e
$relu
%tanh
&softmax
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
ß
-filters
	.sizes
/strides
	0convs
1relu
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_model
»

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
»

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Hkernel
Ibias
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer

r0
s1
t2
u3
v4
w5
x6
y7
z8
{9
|10
}11
~12
13
14
15
16
17
18
819
920
@21
A22
H23
I24
P25
Q26
X27
Y28"
trackable_list_wrapper

r0
s1
t2
u3
v4
w5
x6
y7
z8
{9
|10
}11
~12
13
14
15
16
17
18
819
920
@21
A22
H23
I24
P25
Q26
X27
Y28"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
¾2»
*__inference_conv_att_layer_call_fn_1630193
*__inference_conv_att_layer_call_fn_1630259à
×²Ó
FullArgSpecU
argsMJ
jself
jx_past
j	x_forward
jpos_enc_past
jpos_enc_fwd

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ô2ñ
E__inference_conv_att_layer_call_and_return_conditional_losses_1630674
E__inference_conv_att_layer_call_and_return_conditional_losses_1631089à
×²Ó
FullArgSpecU
argsMJ
jself
jx_past
j	x_forward
jpos_enc_past
jpos_enc_fwd

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
äBá
"__inference__wrapped_model_1628643args_0args_1args_2args_3"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
J
r0
s1
t2
u3
v4
w5"
trackable_list_wrapper
J
r0
s1
t2
u3
v4
w5"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
å2â
*__inference_enc_past_layer_call_fn_1628748
*__inference_enc_past_layer_call_fn_1631174
*__inference_enc_past_layer_call_fn_1631191
*__inference_enc_past_layer_call_fn_1628867¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ2Î
E__inference_enc_past_layer_call_and_return_conditional_losses_1631231
E__inference_enc_past_layer_call_and_return_conditional_losses_1631271
E__inference_enc_past_layer_call_and_return_conditional_losses_1628889
E__inference_enc_past_layer_call_and_return_conditional_losses_1628911¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Á

xkernel
ybias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

zkernel
{bias
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"
_tf_keras_layer
·

|kernel
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"
_tf_keras_layer
·

}kernel
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"
_tf_keras_layer
·

~kernel
±	variables
²trainable_variables
³regularization_losses
´	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"
_tf_keras_layer
«
·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"
_tf_keras_layer
«
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"
_tf_keras_layer
Q
x0
y1
z2
{3
|4
}5
~6"
trackable_list_wrapper
Q
x0
y1
z2
{3
|4
}5
~6"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
¢2
-__inference_spatial_att_layer_call_fn_1631291
-__inference_spatial_att_layer_call_fn_1631311¾
µ²±
FullArgSpec3
args+(
jself
jquery
	jfeature

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_spatial_att_layer_call_and_return_conditional_losses_1631445
H__inference_spatial_att_layer_call_and_return_conditional_losses_1631579¾
µ²±
FullArgSpec3
args+(
jself
jquery
	jfeature

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
8
Î0
Ï1
Ð2"
trackable_list_wrapper
«
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
O
0
1
2
3
4
5"
trackable_list_wrapper
O
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
²
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
ñ2î
-__inference_enc_forward_layer_call_fn_1629082
-__inference_enc_forward_layer_call_fn_1631596
-__inference_enc_forward_layer_call_fn_1631613
-__inference_enc_forward_layer_call_fn_1629201¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ý2Ú
H__inference_enc_forward_layer_call_and_return_conditional_losses_1631719
H__inference_enc_forward_layer_call_and_return_conditional_losses_1631825
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629223
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629245¯
¦²¢
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
,:*	À  2conv_att/dense_5/kernel
%:#  2conv_att/dense_5/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_5_layer_call_fn_1631834¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_5_layer_call_and_return_conditional_losses_1631844¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:)$  2conv_att/dense_6/kernel
%:#  2conv_att/dense_6/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_6_layer_call_fn_1631853¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_6_layer_call_and_return_conditional_losses_1631883¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,:*	À  2conv_att/dense_7/kernel
%:#  2conv_att/dense_7/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_7_layer_call_fn_1631892¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_7_layer_call_and_return_conditional_losses_1631922¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:)@  2conv_att/dense_8/kernel
%:#  2conv_att/dense_8/bias
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_8_layer_call_fn_1631931¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_8_layer_call_and_return_conditional_losses_1631961¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
+:)  2conv_att/dense_9/kernel
%:# 2conv_att/dense_9/bias
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_9_layer_call_fn_1631970¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_9_layer_call_and_return_conditional_losses_1632000¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
þ2û
)__inference_re_lu_3_layer_call_fn_1632005
)__inference_re_lu_3_layer_call_fn_1632010¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1632015
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1632020¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_activation_1_layer_call_fn_1632025¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_activation_1_layer_call_and_return_conditional_losses_1632030¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ÿnon_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_flatten_layer_call_fn_1632035¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_flatten_layer_call_and_return_conditional_losses_1632041¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
7:5  2conv_att/enc_past/conv1d/kernel
-:+  2conv_att/enc_past/conv1d/bias
9:7   2!conv_att/enc_past/conv1d_1/kernel
/:-  2conv_att/enc_past/conv1d_1/bias
9:7   2!conv_att/enc_past/conv1d_2/kernel
/:-  2conv_att/enc_past/conv1d_2/bias
5:3 2!conv_att/spatial_att/dense/kernel
/:- 2conv_att/spatial_att/dense/bias
7:5 2#conv_att/spatial_att/dense_1/kernel
1:/ 2!conv_att/spatial_att/dense_1/bias
7:5  2#conv_att/spatial_att/dense_2/kernel
7:5 2#conv_att/spatial_att/dense_3/kernel
7:5 2#conv_att/spatial_att/dense_4/kernel
<::  2$conv_att/enc_forward/conv1d_3/kernel
2:0  2"conv_att/enc_forward/conv1d_3/bias
<::   2$conv_att/enc_forward/conv1d_4/kernel
2:0  2"conv_att/enc_forward/conv1d_4/bias
<::   2$conv_att/enc_forward/conv1d_5/kernel
2:0  2"conv_att/enc_forward/conv1d_5/bias
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
áBÞ
%__inference_signature_wrapper_1631157args_0args_1args_2args_3"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Á

rkernel
sbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

tkernel
ubias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

vkernel
wbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
£2 
'__inference_re_lu_layer_call_fn_1632046
'__inference_re_lu_layer_call_fn_1632051
'__inference_re_lu_layer_call_fn_1632056¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
B__inference_re_lu_layer_call_and_return_conditional_losses_1632061
B__inference_re_lu_layer_call_and_return_conditional_losses_1632066
B__inference_re_lu_layer_call_and_return_conditional_losses_1632071¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
?
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
	variables
 trainable_variables
¡regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
'
|0"
trackable_list_wrapper
'
|0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
¥	variables
¦trainable_variables
§regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
'
}0"
trackable_list_wrapper
'
}0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
«	variables
¬trainable_variables
­regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
'
~0"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
±	variables
²trainable_variables
³regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
·	variables
¸trainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
X
0
 1
!2
"3
#4
$5
%6
&7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Â

kernel
	bias
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
Ñ	variables
Òtrainable_variables
Óregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
©2¦
)__inference_re_lu_2_layer_call_fn_1632076
)__inference_re_lu_2_layer_call_fn_1632081
)__inference_re_lu_2_layer_call_fn_1632086¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1632091
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1632096
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1632101¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
?
Î0
Ï1
Ð2
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_conv1d_layer_call_fn_1632110¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv1d_layer_call_and_return_conditional_losses_1632125¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv1d_1_layer_call_fn_1632134¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1632149¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv1d_2_layer_call_fn_1632158¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1632173¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/
0
1"
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv1d_3_layer_call_fn_1632182¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1632219¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
É	variables
Êtrainable_variables
Ëregularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv1d_4_layer_call_fn_1632228¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1632265¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_conv1d_5_layer_call_fn_1632274¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1632311¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper¯
"__inference__wrapped_model_1628643"rstuvw89@Axyz{}|~HIPQXY¨¢¤
¢
$!
args_0ÿÿÿÿÿÿÿÿÿ
$!
args_1ÿÿÿÿÿÿÿÿÿd
$!
args_2ÿÿÿÿÿÿÿÿÿ

$!
args_3ÿÿÿÿÿÿÿÿÿd
ª "7ª4
2
output_1&#
output_1ÿÿÿÿÿÿÿÿÿ
­
I__inference_activation_1_layer_call_and_return_conditional_losses_1632030`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ

ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 
.__inference_activation_1_layer_call_fn_1632025S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
­
E__inference_conv1d_1_layer_call_and_return_conditional_losses_1632149dtu3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_conv1d_1_layer_call_fn_1632134Wtu3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ­
E__inference_conv1d_2_layer_call_and_return_conditional_losses_1632173dvw3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_conv1d_2_layer_call_fn_1632158Wvw3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¶
E__inference_conv1d_3_layer_call_and_return_conditional_losses_1632219m7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
d
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
b 
 
*__inference_conv1d_3_layer_call_fn_1632182`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
d
ª " ÿÿÿÿÿÿÿÿÿ
b ·
E__inference_conv1d_4_layer_call_and_return_conditional_losses_1632265n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
b 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
` 
 
*__inference_conv1d_4_layer_call_fn_1632228a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
b 
ª " ÿÿÿÿÿÿÿÿÿ
` ·
E__inference_conv1d_5_layer_call_and_return_conditional_losses_1632311n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
` 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
^ 
 
*__inference_conv1d_5_layer_call_fn_1632274a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
` 
ª " ÿÿÿÿÿÿÿÿÿ
^ «
C__inference_conv1d_layer_call_and_return_conditional_losses_1632125drs3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 
(__inference_conv1d_layer_call_fn_1632110Wrs3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ Ö
E__inference_conv_att_layer_call_and_return_conditional_losses_1630674"rstuvw89@Axyz{}|~HIPQXYº¢¶
®¢ª
$!
x_pastÿÿÿÿÿÿÿÿÿ
'$
	x_forwardÿÿÿÿÿÿÿÿÿd
*'
pos_enc_pastÿÿÿÿÿÿÿÿÿ

)&
pos_enc_fwdÿÿÿÿÿÿÿÿÿd
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 Ö
E__inference_conv_att_layer_call_and_return_conditional_losses_1631089"rstuvw89@Axyz{}|~HIPQXYº¢¶
®¢ª
$!
x_pastÿÿÿÿÿÿÿÿÿ
'$
	x_forwardÿÿÿÿÿÿÿÿÿd
*'
pos_enc_pastÿÿÿÿÿÿÿÿÿ

)&
pos_enc_fwdÿÿÿÿÿÿÿÿÿd
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 ®
*__inference_conv_att_layer_call_fn_1630193ÿ"rstuvw89@Axyz{}|~HIPQXYº¢¶
®¢ª
$!
x_pastÿÿÿÿÿÿÿÿÿ
'$
	x_forwardÿÿÿÿÿÿÿÿÿd
*'
pos_enc_pastÿÿÿÿÿÿÿÿÿ

)&
pos_enc_fwdÿÿÿÿÿÿÿÿÿd
p 
ª "ÿÿÿÿÿÿÿÿÿ
®
*__inference_conv_att_layer_call_fn_1630259ÿ"rstuvw89@Axyz{}|~HIPQXYº¢¶
®¢ª
$!
x_pastÿÿÿÿÿÿÿÿÿ
'$
	x_forwardÿÿÿÿÿÿÿÿÿd
*'
pos_enc_pastÿÿÿÿÿÿÿÿÿ

)&
pos_enc_fwdÿÿÿÿÿÿÿÿÿd
p
ª "ÿÿÿÿÿÿÿÿÿ
¥
D__inference_dense_5_layer_call_and_return_conditional_losses_1631844]890¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
)__inference_dense_5_layer_call_fn_1631834P890¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ
ª "ÿÿÿÿÿÿÿÿÿ ¬
D__inference_dense_6_layer_call_and_return_conditional_losses_1631883d@A3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
$
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
 
)__inference_dense_6_layer_call_fn_1631853W@A3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
$
ª "ÿÿÿÿÿÿÿÿÿ
 ­
D__inference_dense_7_layer_call_and_return_conditional_losses_1631922eHI4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
À
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
 
)__inference_dense_7_layer_call_fn_1631892XHI4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
À
ª "ÿÿÿÿÿÿÿÿÿ
 ¬
D__inference_dense_8_layer_call_and_return_conditional_losses_1631961dPQ3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
@
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
 
)__inference_dense_8_layer_call_fn_1631931WPQ3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
@
ª "ÿÿÿÿÿÿÿÿÿ
 ¬
D__inference_dense_9_layer_call_and_return_conditional_losses_1632000dXY3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ

 
)__inference_dense_9_layer_call_fn_1631970WXY3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
 
ª "ÿÿÿÿÿÿÿÿÿ
Æ
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629223z<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
d
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
^ 
 Æ
H__inference_enc_forward_layer_call_and_return_conditional_losses_1629245z<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
d
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
^ 
 À
H__inference_enc_forward_layer_call_and_return_conditional_losses_1631719t6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ
d
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
^ 
 À
H__inference_enc_forward_layer_call_and_return_conditional_losses_1631825t6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ
d
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
^ 
 
-__inference_enc_forward_layer_call_fn_1629082m<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
d
p 
ª " ÿÿÿÿÿÿÿÿÿ
^ 
-__inference_enc_forward_layer_call_fn_1629201m<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
d
p
ª " ÿÿÿÿÿÿÿÿÿ
^ 
-__inference_enc_forward_layer_call_fn_1631596g6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ
d
p 
ª " ÿÿÿÿÿÿÿÿÿ
^ 
-__inference_enc_forward_layer_call_fn_1631613g6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ
d
p
ª " ÿÿÿÿÿÿÿÿÿ
^ ¶
E__inference_enc_past_layer_call_and_return_conditional_losses_1628889mrstuvw8¢5
.¢+
%"
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 ¶
E__inference_enc_past_layer_call_and_return_conditional_losses_1628911mrstuvw8¢5
.¢+
%"
input_1ÿÿÿÿÿÿÿÿÿ
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 °
E__inference_enc_past_layer_call_and_return_conditional_losses_1631231grstuvw2¢/
(¢%

xÿÿÿÿÿÿÿÿÿ
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 °
E__inference_enc_past_layer_call_and_return_conditional_losses_1631271grstuvw2¢/
(¢%

xÿÿÿÿÿÿÿÿÿ
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_enc_past_layer_call_fn_1628748`rstuvw8¢5
.¢+
%"
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ 
*__inference_enc_past_layer_call_fn_1628867`rstuvw8¢5
.¢+
%"
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ 
*__inference_enc_past_layer_call_fn_1631174Zrstuvw2¢/
(¢%

xÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ 
*__inference_enc_past_layer_call_fn_1631191Zrstuvw2¢/
(¢%

xÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ ¥
D__inference_flatten_layer_call_and_return_conditional_losses_1632041]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ
 }
)__inference_flatten_layer_call_fn_1632035P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿÀ°
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1632091h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
b 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
b 
 °
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1632096h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
` 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
` 
 °
D__inference_re_lu_2_layer_call_and_return_conditional_losses_1632101h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
^ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
^ 
 
)__inference_re_lu_2_layer_call_fn_1632076[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
^ 
ª " ÿÿÿÿÿÿÿÿÿ
^ 
)__inference_re_lu_2_layer_call_fn_1632081[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
` 
ª " ÿÿÿÿÿÿÿÿÿ
` 
)__inference_re_lu_2_layer_call_fn_1632086[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
b 
ª " ÿÿÿÿÿÿÿÿÿ
b  
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1632015X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ¨
D__inference_re_lu_3_layer_call_and_return_conditional_losses_1632020`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
 
)__inference_re_lu_3_layer_call_fn_1632005S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
 
ª "ÿÿÿÿÿÿÿÿÿ
 x
)__inference_re_lu_3_layer_call_fn_1632010K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¦
B__inference_re_lu_layer_call_and_return_conditional_losses_1632061`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 ¦
B__inference_re_lu_layer_call_and_return_conditional_losses_1632066`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 ¦
B__inference_re_lu_layer_call_and_return_conditional_losses_1632071`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 ~
'__inference_re_lu_layer_call_fn_1632046S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ~
'__inference_re_lu_layer_call_fn_1632051S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ~
'__inference_re_lu_layer_call_fn_1632056S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ Ú
%__inference_signature_wrapper_1631157°"rstuvw89@Axyz{}|~HIPQXYÐ¢Ì
¢ 
ÄªÀ
.
args_0$!
args_0ÿÿÿÿÿÿÿÿÿ
.
args_1$!
args_1ÿÿÿÿÿÿÿÿÿd
.
args_2$!
args_2ÿÿÿÿÿÿÿÿÿ

.
args_3$!
args_3ÿÿÿÿÿÿÿÿÿd"7ª4
2
output_1&#
output_1ÿÿÿÿÿÿÿÿÿ
ä
H__inference_spatial_att_layer_call_and_return_conditional_losses_1631445xyz{}|~a¢^
W¢T
# 
queryÿÿÿÿÿÿÿÿÿ
 
)&
featureÿÿÿÿÿÿÿÿÿd
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
d
 ä
H__inference_spatial_att_layer_call_and_return_conditional_losses_1631579xyz{}|~a¢^
W¢T
# 
queryÿÿÿÿÿÿÿÿÿ
 
)&
featureÿÿÿÿÿÿÿÿÿd
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
d
 ¼
-__inference_spatial_att_layer_call_fn_1631291xyz{}|~a¢^
W¢T
# 
queryÿÿÿÿÿÿÿÿÿ
 
)&
featureÿÿÿÿÿÿÿÿÿd
p 
ª "ÿÿÿÿÿÿÿÿÿ
d¼
-__inference_spatial_att_layer_call_fn_1631311xyz{}|~a¢^
W¢T
# 
queryÿÿÿÿÿÿÿÿÿ
 
)&
featureÿÿÿÿÿÿÿÿÿd
p
ª "ÿÿÿÿÿÿÿÿÿ
d