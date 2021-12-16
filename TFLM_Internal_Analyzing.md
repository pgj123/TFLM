# 문서 기술의 목적

TensorFlow 공식 문서를 통해 마이크로컨트롤러 기반 추론 환경을 TFLM을 통해 조성하는 방법론에 대한 내용은 공식적으로 문서화되어 있지만, 런타임 시점부터 추론을 수행하기 위한 준비과정까지를 실행 흐름에 따라 내부적으로 어떠한 접근이 일어나는 지를 설명한 문서는 없었다.

따라서 본 문서를 통하여 TFLM의 추론을 수행하기 위한 준비과정의 내부 구조 흐름를 기술하고 독자들에게 도움을 제공하고자 한다.

***

## 목차

1. [TFLM 라이브러리](#TFLM-라이브러리)
2. [FlatBuffer](#FlatBuffer)
3. [Tensor Arena](#Tensor-Arena)
   1. [Head Section](#Head-Section)
   2. [Temp Section](#Temp-Section)
   3. [Tail Section](#Tail-Section)
4. [Setup](#Setup)
   1. [Flatbuffer model 불러오기](#Flatbuffer-model-불러오기)
   2. [Operations resolver 선언](#Operations-resolver-선언)
   3. [Tensor arena 메모리 할당](#Tensor-arena-메모리-할당)
   4. [Interpreter 인스턴스 생성](#Interpreter-인스턴스-생성)
   5. [Interpreter에 Tensor](#Interpreter에-Tensor들을-할당)
5. [Allocate_Tensors](#allocate_tensors)
   1. [StartModelAllocation](#StartModelAllocation)
   2. [SetSubgraphAllocations](#SetSubgraphAllocations)
   3. [PrepareNodeAndRegistrationDataFromFlatbuffer](#PrepareNodeAndRegistrationDataFromFlatbuffer)
   4. [InitSubgraphs](#InitSubgraphs)
   5. [PrepareSubgraphs](#PrepareSubgraphs)
   6. [FinishModelAllocation](#FinishModelAllocation)
   7. [AllocatePersistentBuffer & AllocatePersistentTfLiteTensor](#allocatepersistentbuffer-allocatepersistenttfLitetensor)
   8. [ResetVariableTensors](#ResetVariableTensors)

***




# TFLM 라이브러리

TFLM C++ 라이브러리는 기본적으로 TensorFlow Lite와 호환되도록 설계되었으며 마이크로컨트롤러 한정 라이브러리로서 제한된 기능으로 추가나 삭제 또는 재정의된다.

마이크로컨트롤러용 헤더파일들은 tflite-micro 공식 깃헙 문서를 바탕으로 tflite-micro/tensorflow/lite/micro 폴더 위치에 선언되어 있고 대표적인 헤더파일들의 종류 및 역할은 TensorFlow 공식 문서를 참조한 바에 따라 아래와 같다.

![스크린샷, 2021-12-15 19-36-41](https://user-images.githubusercontent.com/76988777/146171164-e422cf9e-e0c4-4762-abe8-ada151421e8d.png)

***

# FlatBuffer

![스크린샷, 2021-12-15 14-04-48](https://user-images.githubusercontent.com/76988777/146126800-05246d8c-6037-4630-9eda-c1774d478cdc.png)

마이크로컨트롤러 보드에서 모델 추론을 수행하기 위함은 TFLM 라이브러리를 사용하는 목적 중 하나이다. 마이크로컨트롤러 보드는 한정된 용량의 SRAM 및 FLASH 메모리를 가지기 때문에 메모리 사용에 있어 주의를 기울여야 한다. 

뿐만 아니라 모델의 구조가 복잡한, 학습된 DNN 모델을 수행하기 위해서는 방대한 양의 연산을 처리해야 하며 학습이 완료된 후 결과물로써 저장된 파일의 용량 역시 상대적으로 크기 때문에 마이크로컨트롤러 기반 추론 환경에 있어 그대로 사용하는 데엔 무리가 있다. 

그렇기 때문에 마이크로컨트롤러 환경에서 실행될 수 있도록 학습된 모델 파일을 적절한 형식인 FlatBuffer로 변환해주어 모델 크기를 줄이고 Tensorflow Lite가 제한적인 연산만을 사용하도록 수정해야 한다(제한된 수행가능 연산 리스트는 all_ops_resolver.cc 파일을 통해 확인할 수 있다.). 

***

FlatBuffer는 구글에서 개발된 크로스 플랫폼 직렬화 라이브러리로, FlatBuffer를 사용함으로써 메모리 효율이 증가하고 속도가 증가하며, 패킹/언 패킹 없이 직렬화 된 데이터에 엑세스를 할 수 있다. 데이터가 직렬화 되었다는 것은 객체의 내용을 바이트 단위로 변환하여 입출력에 구애받지 않고 파일 또는 네트워크를 통해서 송수신이 가능토록 하는 것을 의미한다. 

변환 과정을 통해 생성된 코드는 작으며 단일 header 파일로 통합이 쉬워진다. FlatBuffer를 생성하기 위해서는 구조를 담은 스키마를 작성해야 하며, 해당 내용이 TFlite에서 작성된 파일은 schema.fbs 파일을 통해 참조할 수 있다.

타겟 보드의 제약에 맞는 적절한 모델을 찾는 것 역시 필요하지만, 이 이전에 학습된 모델 파일의 FlatBuffer로의 변환은 마이크로컨트롤러 환경에서의 추론을 준비함에 있어 필수적 요소이다. 

***
# Tensor Arena

![image](https://user-images.githubusercontent.com/76988777/146195973-ca2ab75c-d9b6-4176-96dc-7a6d8726bfc4.png)

위 그림은 TFLM의 Arena 영역으로, SRAM 메모리 상에 위치하며 해당 공간을 포함한 충분한(또는 적절한) 영역의 크기를 사용자가 직접 선언하여 정의된다. Tensor Arena라는 공간은 다음과 같은 목적을 수행하기 위해 정의된다.

+ 입력, 출력, 중간 결과 배열들을 보관하기 위하여 사용
+ 추론 과정에 있어 읽기가 수행되며 동시에 추론 완료 시 까지 유지되어야 하는 영구적인 특성을 갖는 버퍼를 보관하기 위하여 사용
+ 해당 공간을 공유함으로써 수동 메모리 관리를 효율적으로 수행하기 위하여 사용


***

또한 Arena 영역은 Head section, Temp section, Tail section의 이름을 갖는 논리적인 세 영역으로 구분되는데, Tail section은 지정된 arena 주소의 끝에서부터 공간이 할당됨에 따라 arena의 시작주소 방향으로 확장되고, Head section은 반대로 시작주소에서 증가하는 방향으로 확장된다. 따라서 추론 과정 중 arena 영역의 공간 부족 현상은 Head section과 Tail section이 교차하는 상황이 발생하는 경우에 일어난다. Temp section은 Head section의 끝 주소에서부터 시작되며, arena의 마지막 주소를 향해 증가하는 방향으로 확장된다.






### Head Section

추론과정이 완료되는 시점까지 계속 보관되어야 할 필요가 없는, non-persistent한 특성을 지닌 버퍼들이 저장되는 영역이다. 대표적으로 CMSIS-NN과 같은 뉴럴 네트워크 추론에 최적화된 함수들이 선언된 헤더 파일을 참조하여 추론에 있어 사용될 함수들을 골라(scratch 하여) 버퍼(scratch buffer)에 저장된다.

### Temp Section

Head Section와 비교하여 더 짧은 life-cycle을 갖는, temporary한 특성의 데이터들이 저장되는 영역이다. Head Section의 끝 주소를 Temp Section의 시작 주소로 갖기 때문에 Head Section의 확장 및 축소가 일어나기 전 반드시 Temp Section의 리셋 과정이 선행되어야 한다. 대개 한 메소드를 수행하는 시간 정도의 생명 주기를 갖는 일시 데이터들이 저장되는 영역이다.

### Tail Section

Arena 영역이 존재하는 생명 주기 동안 영구적으로 저장되는 값들과 구조체들이 모두 저장되는 영역이다. 추론 준비 단계에서 FlatBuffer의 양자화 수준과 관련된 정보를 저장하기도 하며, Node & Registration 정보 및 메모리 사용 로그를 기록하는 recording API와 관련된 정보, 입출력 텐서 등 다양한 데이터들이 저장된다.





***
# Setup
TFLM 라이브러리를 사용하여 마이크로컨트롤러를 통해 학습한 모델을 추론하고자 한다. 이때 추론을 하기 전 TFLM에서 정의된, 정형화된 절차를 통해 추후 모델 추론 과정을 성공적으로 수행하기 위한 준비 단계를 거치게 된다. 해당 과정 중 AllocateTensors() 라는 함수가 불리기까지의 과정을 묶어 setup 과정이라고 부르자. TFLM에서 setup 과정은 다음 단계들을 수행한다.



### Flatbuffer model 불러오기

![스크린샷, 2021-12-15 19-42-43](https://user-images.githubusercontent.com/76988777/146172128-c972d4a9-99de-42fc-bfaa-465ea67e05c3.png)

위 과정을 통해 직렬화된 char 배열인 cifar10_lenet_original_no_quant라는 데이터를 로드하여(역직렬화 하여) 인스턴스화 한다. 이후 모델에서 스키마 버전이 사용 중인 버전화 호환되는지를 확인하는 절차를 수행한다.

***

### Operations resolver 선언

![스크린샷, 2021-12-15 19-53-54](https://user-images.githubusercontent.com/76988777/146173883-3a458826-b35f-4b87-9448-653c2b61f21f.png)

AllOpsResolver는 마이크로컨트롤러용 TensorFlow Lite에서 사용할 수 있는 모든 연산을 로드하며, 여기에 많은 메모리가 사용된다. 특정 모델은 이러한 연산의 일부만 사용하므로 실제 어플리케이션에서는 필요한 연산만 로드하는 것이 좋다.
***


### Tensor arena 메모리 할당

![스크린샷, 2021-12-15 19-56-26](https://user-images.githubusercontent.com/76988777/146174241-1845a5db-0146-4c72-9582-508b2a1e0302.png)

입력, 출력, 및 중간 layer 결과값 저장을 위한 배열에 대해 일정량의 메모리를 미리 할당해야 한다. 이 메모리는 tensor_arena_size 크기의 uint8_t 배열로 제공됩니다. 사용 중인 보드가 갖는 SRAM 용량 크기 및 사용하려는 모델의 크기를 함께 고려하여 실험적으로 적절한 값을 찾아야 한다. 
***

### Interpreter 인스턴스 생성

![스크린샷, 2021-12-15 20-04-06](https://user-images.githubusercontent.com/76988777/146175318-f3e1d909-e636-442c-a954-2fb0ca942bd5.png)

tflite::MicroInterpreter 인스턴스를 만들고 앞서 만든 변수를 전달한다.
***
### Interpreter에 tensor들을 할당

![스크린샷, 2021-12-15 20-05-30](https://user-images.githubusercontent.com/76988777/146175588-4fcaaae7-5fcf-4567-9551-d9aeecbeb518.png)

앞서 인터프리터에 지정된 모델의 tensor들을 tensor_arena 공간에 할당한다.
***
_여기까지 직렬화된 FlatBuffer 모델을 사용하는 이유부터 Allocate_Tensors() 함수가 불리기 직전까지의 과정을 살펴보았다._    
***
# Allocate_Tensors

본 메소드는 micro_interpreter.h에 선언된 MicroInterpreter class에 속해  있는 메소드로 아래와 같은 순서로 수행된다.

![스크린샷, 2021-12-15 20-35-39](https://user-images.githubusercontent.com/76988777/146179567-d7276e06-51c6-4284-b8ea-6ccf0aa49c25.png)
***
### StartModelAllocation

![image](https://user-images.githubusercontent.com/76988777/146357415-1963864d-da90-4b82-aeca-b6f367175a58.png)


+ Tail Section에 메모리를 할당하여 MicroBuiltInDataAllocator를 Interpreter에 최초 매핑하는 과정이다.

![image](https://user-images.githubusercontent.com/76988777/146357519-11666048-aae9-4ea8-97f5-2a38ef3fea2f.png)

+ 연산이 수행되는 tensor들을 할당해주는 SubgraphAllocations의 이름을 갖는 구조체가 존재하며 해당 역할을 수행하기 위한 구조체를 할당한다.

![image](https://user-images.githubusercontent.com/76988777/146358087-01a8f6a6-402b-4d3b-a497-8562e09c0c15.png)

+ 앞서 할당한 SubgraphAllocation 구조체의 멤버인 NodeAndRegistration을 사용하기 위한 공간 역시 할당한 후, subgraph_allocations에 매핑한다. 다른 멤버인 TfLiteEvalTensor 역시 동일한 과정을 수행하고,  StartModelAllocation 메소드의 반환형으로써 공간 할당이 완료된 SubgraphAllocations 구조체를 반환한다.

***

### SetSubgraphAllocations

![image](https://user-images.githubusercontent.com/76988777/146359004-3e7d6c18-4fdd-4b81-847d-68a16aeb8390.png)

+ StartModelAllocation 결과 생성된 SubgraphAllocation을 graph에 매핑한다. 현재 참조되는 graph의 scope는 MicroInterpreter::AllocateTensors에 속해 있으며 MicroInterpreter는 멤버 변수로 graph를 지니고 있다.

![image](https://user-images.githubusercontent.com/76988777/146360031-ec6c0364-90b5-4d76-aaf7-505190ef7002.png)


***

### PrepareNodeAndRegistrationDataFromFlatbuffer

![image](https://user-images.githubusercontent.com/76988777/146232902-c33aff9e-7254-4c7e-b9c8-ed6f2d48ae54.png)

+ FlatBuffer에 있는 학습된 모델과 관련된 메타데이터를 참조하여 operation option type을 판단하는 부분이다. 제한된 개수로 지원되는 내장형 operation option type에 해당되는 경우와, FlatBuffer로 변환 간 사용자 지정 operation type 지정 함수를 사용한 결과가 반영된 custom option type인 경우를 구분한다.
***
![image](https://user-images.githubusercontent.com/76988777/146234530-1ade4c44-bcb2-4221-b0fa-0644b4a57b93.png)

+ interpreter 인스턴스에 매핑되어 있는 op_resolver 멤버를 이용하여 현재 시점에서 확인된 operation type을 어떤 함수로 파싱할지를 판단한다. 현재 시점이란, 본 PrepareNodeAndRegistrationDataFromFlatbuffer 함수에서 모든 opeartion들에 대해 반복문을 돌며 수행되는 과정 중 해당되는 opeartion과 관련된 시점을 의미한다.
***
![image](https://user-images.githubusercontent.com/76988777/146235155-8754e92f-cc14-45d0-bbdf-43ceada8b8db.png)

+ FlatBuffer로부터 node의 입출력 데이터 및 내장/사용자정의 데이터를 지정된 node 변수의 멤버로 매핑한다.
***
### InitSubgraphs

![image](https://user-images.githubusercontent.com/76988777/146235628-16cda086-06c8-4222-9213-f31a56eb3c06.png)

+ PrepareNodeAndRegistrationDataFromFlatbuffer에서 매핑이 완료된 내장/사용자정의 데이터를 불러와서 node 별 상황에 맞는 user_data를 받을 수 있도록 초기화한다.
***
### PrepareSubgraphs



+ 다시 node와 registration 값들을 불러온 후, graph가 사용할 준비가 완료되었는지 체크하는 과정으로 통과되면 최종 allocation이 완료되었다고 판단이 가능하다.

***
### FinishModelAllocation

![image](https://user-images.githubusercontent.com/76988777/146242180-61c0faf6-9015-427d-9eb4-999261f7ad2f.png)

+ 이 메소드는 앞서 설명한 StartModelAllocation 메소드가 수행된 이후 출력되도록 하여 SubgraphAllocations 구조체가 업데이트 된 상태의 입력인자를 가진 채로 호출되어야 한다. 
+ 입력인자로 받는 ScratchBufferHandle의 정보가 갱신되며 현재 subgraph index 위치에 저장되어 있는 tensor들의 정보들을 바탕으로 메모리 사용 계획을 수립한다. 
+ 이때, 계획된 정보는 AllocationInfo의 이름을 가진 구조체에 저장된다.
+ 또한, 수립된 계획에 따라 ScratchBufferHandle에 요청할 시점들도 기록한다. 
+ CommitStaticMemoryPlan 메소드 수행 간 AllocationInfoBuilder 구조체의 도움을 받으며 최종적으로 결정될 Head Section 영역 및 Temp Section 영역은 반복문을 수행하며 가장 큰 공간을 차지하는 Head Section에 의해 결정된다.
+ 이를 통해 계획된 offset들을 바탕으로 eval tensor들을 갱신한다. 
+ 갱신 간 앞서 subgraph의 정보가 variable이었어서 memset 되었던 영역들을 이 시점에서 재할당한다.
+ ScratchBufferHandle 구조체에 저장된 계획안은 차후 실사용 시점에 GetScratchBuffer 메소드를 통해 활용된다.


***
### AllocatePersistentBuffer AllocatePersistentTfLiteTensor

![image](https://user-images.githubusercontent.com/76988777/146240218-dd5f26c1-9136-4081-bfbb-5d8d641aea3e.png)

+ input tensor와 output tensor를 위한 메모리를 Tail Section에 할당한다. 위 사진은 해당 과정 중 input tensor를 할당하는 과정이다.
***
### ResetVariableTensors

![image](https://user-images.githubusercontent.com/76988777/146360183-2cc03eb1-3171-46d8-b570-1700d13d3404.png)

+ 여태 셋팅한 subgraph 정보 중 tensor들을 for문을 돌면서 확인하여, 모델에서 variable tensor에 해당하는 경우들을 모두 찾아, 0으로 memset(zeros out) 처리한다.
***



참고자료 : https://www.tensorflow.org/lite/
