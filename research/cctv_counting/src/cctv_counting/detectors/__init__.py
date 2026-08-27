"""detector 구현들.

각 구현은 자기가 필요한 무거운 라이브러리를 자기 모듈 안에서 import한다.
그래서 여기서 하위 모듈을 미리 끌어오지 않는다. torch가 없는 환경에서
cctv_counting을 import하는 것만으로 터지면 안 된다.
"""
