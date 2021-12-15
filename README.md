# TFLM 라이브러리

TFLM C++ 라이브러리는 TensorFlow Lite와 호환되도록 설계되어 기본적으로는 TFLite에서 정의된 구조를 통해 접근하되 마이크로컨트롤러 한정 라이브러리를 사용하는 시점에 참조된다.
마이크로컨트롤러용 헤더파일들의 종류 및 역할은 TensorFlow 공식 문서를 참조한 바에 따라 아래와 같다.

![스크린샷, 2021-12-15 19-36-41](https://user-images.githubusercontent.com/76988777/146171164-e422cf9e-e0c4-4762-abe8-ada151421e8d.png)


# FlatBuffer

![스크린샷, 2021-12-15 14-04-48](https://user-images.githubusercontent.com/76988777/146126800-05246d8c-6037-4630-9eda-c1774d478cdc.png)

마이크로컨트롤러 보드에서 모델 추론을 수행하기 위함은 TFLM 라이브러리를 사용하는 목적 중 하나이다. 마이크로컨트롤러 보드는 한정된 용량의 SRAM 및 FLASH 메모리를 가지기 때문에 메모리 사용에 있어 주의를 기울여야 한다. 뿐만 아니라 모델의 구조가 복잡한, 학습된 DNN 모델을 수행하기 위해서는 방대한 양의 연산을 처리해야 하며 학습이 완료된 후 결과물로써 저장된 파일의 용량 역시 상대적으로 크기 때문에 마이크로컨트롤러 기반 추론 환경에 있어 그대로 사용하는 데엔 무리가 있다. 

그렇기 때문에 마이크로컨트롤러 환경에서 실행될 수 있도록 학습된 모델 파일을 적절한 형식인 FlatBuffer로 변환해주어 모델 크기를 줄이고 Tensorflow Lite가 제한적인 연산만을 사용하도록 수정해야 한다(제한된 수행가능 연산 리스트는 all_ops_resolver.cc 파일을 통해 확인할 수 있다.). 

FlatBuffer는 구글에서 개발된 크로스 플랫폼 직렬화 라이브러리로, FlatBuffer를 사용함으로써 메모리 효율이 증가하고 속도가 증가하며, 패킹/언 패킹 없이 직렬화 된 데이터에 엑세스를 할 수 있다. 데이터가 직렬화 되었다는 것은 객체의 내용을 바이트 단위로 변환하여 입출력에 구애받지 않고 파일 또는 네트워크를 통해서 송수신이 가능토록 하는 것을 의미한다. 

변환 과정을 통해 생성된 코드는 작으며 단일 header 파일로 통합이 쉬워진다. FlatBuffer를 생성하기 위해서는 구조를 담은 스키마를 작성해야 하며, 해당 내용이 TFlite에서 작성된 파일은 schema.fbs 파일을 통해 참조할 수 있다.

타겟 보드의 제약에 맞는 적절한 모델을 찾는 것 역시 필요하지만, 이 이전에 학습된 모델 파일의 FlatBuffer로의 변환은 마이크로컨트롤러 환경에서의 추론을 준비함에 있어 필수적 요소이다. 

# Overall Structure of this Document

이 문서에서는 학습한 모델이 로드되고, 추론이 일어나기 전 추론 수행을 위한 준비 과정을 실행 순서대로 살펴볼 것이다. 이후, 추론 수행이 실제로 일어나는 과정을 내부적인 흐름에 따라 살펴볼 것이다.





# Setup
TFLM 라이브러리를 사용하여 마이크로컨트롤러를 통해 학습한 모델을 추론하고자 한다. 이때 추론을 하기 전 TFLM에서 정의된, 정형화된 절차를 통해 추후 모델 추론 과정을 성공적으로 수행하기 위한 준비 단계를 거치게 된다. 해당 과정 중 AllocateTensors() 라는 함수가 불리기까지의 과정을 묶어 setup 과정이라고 부르자. TFLM에서 setup 과정은 다음 단계들을 수행한다.



### 1. Flatbuffer model 불러오기

![스크린샷, 2021-12-15 19-42-43](https://user-images.githubusercontent.com/76988777/146172128-c972d4a9-99de-42fc-bfaa-465ea67e05c3.png)

위 과정을 통해 직렬화된 char 배열인 cifar10_lenet_original_no_quant라는 데이터를 로드하여(역직렬화 하여) 인스턴스화 한다. 이후 모델에서 스키마 버전이 사용 중인 버전화 호환되는지를 확인하는 절차를 수행한다.

***

### 2. Operations resolver 선언

![스크린샷, 2021-12-15 19-53-54](https://user-images.githubusercontent.com/76988777/146173883-3a458826-b35f-4b87-9448-653c2b61f21f.png)

AllOpsResolver는 마이크로컨트롤러용 TensorFlow Lite에서 사용할 수 있는 모든 연산을 로드하며, 여기에 많은 메모리가 사용된다. 특정 모델은 이러한 연산의 일부만 사용하므로 실제 어플리케이션에서는 필요한 연산만 로드하는 것이 좋다.
***


### 3. tensor_arena 메모리 할당

![스크린샷, 2021-12-15 19-56-26](https://user-images.githubusercontent.com/76988777/146174241-1845a5db-0146-4c72-9582-508b2a1e0302.png)

입력, 출력, 및 중간 layer 결과값 저장을 위한 배열에 대해 일정량의 메모리를 미리 할당해야 한다. 이 메모리는 tensor_arena_size 크기의 uint8_t 배열로 제공됩니다. 사용 중인 보드가 갖는 SRAM 용량 크기 및 사용하려는 모델의 크기를 함께 고려하여 실험적으로 적절한 값을 찾아야 한다. 
***

### 4. interpreter 인스턴스 생성

![스크린샷, 2021-12-15 20-04-06](https://user-images.githubusercontent.com/76988777/146175318-f3e1d909-e636-442c-a954-2fb0ca942bd5.png)

tflite::MicroInterpreter 인스턴스를 만들고 앞서 만든 변수를 전달한다.
***
### 5. interpreter에 tensor들을 할당

![스크린샷, 2021-12-15 20-05-30](https://user-images.githubusercontent.com/76988777/146175588-4fcaaae7-5fcf-4567-9551-d9aeecbeb518.png)

앞서 인터프리터에 지정된 모델의 tensor들을 tensor_arena 공간에 할당한다.
***
_여기까지 직렬화된 모델을 로드하는 것 부터 Allocate_Tensors() 함수가 불리기까지의 과정을 살펴보았다._    

# Allocate_Tensors()



ddddd





참고자료 : https://www.tensorflow.org/lite/
