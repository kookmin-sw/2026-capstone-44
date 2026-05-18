[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lvs6kcL8)
# 드론 영상 기반 Visual Grounding 성능 개선 연구


## 팀소개 및 페이지를 꾸며주세요.

- 프로젝트 소개
  - 프로젝트 설치방법 및 데모, 사용방법, 프리뷰등을 readme.md에 작성.
  - Api나 사용방법등 내용이 많을경우 wiki에 꾸미고 링크 추가.
  
- 팀페이지 꾸미기
  - 프로젝트 소개 및 팀원 소개
  - index.md 예시보고 수정.

- GitHub Pages 리파지토리 Settings > Options > GitHub Pages 
  - Source를 marster branch
  - Theme Chooser에서 태마선택
  - 수정후 팀페이지 확인하여 점검.

**팀페이지 주소** -> https://kookmin-sw.github.io/ '{{자신의 리파지토리 아이디}}'

**예시)** 2023년 0조  https://kookmin-sw.github.io/capstone-2023-00/


## 내용에 아래와 같은 내용들을 추가하세요.

### 1. 프로젝트 소개

#### 드론 영상 기반 Visual Grounding 성능 개선 연구 ####
[아세아항측 공간정보/원격탐사 전문 기업, 김영욱 교수님]

- 드론(UAV)에서 촬영한 항공 이미지를 대상으로 자연어 설명 문장에 포함된 물체의 bounding box를 출력으로 내는 Aerial Visual Grounding task
- 목표: 항공 이미지는 넓은 시야각으로 인한 작은 객체 크기, 복잡한 공간적 분포 등의 특성을 띄고 있어 해당 특성 하에서 이미지와 텍스트의 관계 학습을 효과적으로 해야 하는 도전 과제를 해결하는 것
- AerialVG (https://arxiv.org/abs/2504.07836) 데이터셋 상에서 기존 baseline 모델 이상의 성능을 내는 모델 방법론 고안 및 검증

### 2. 소개 영상

프로젝트 소개하는 영상을 추가하세요

### 3. 팀 소개

## 👥 Members 👥

<table>
  <tr>
    <td align="center" width="25%">
      <a href="https://github.com/suyamg">
        <img src="https://github.com/suyamg.png" width="130px;" style="border-radius:50%"/>
        <br /><br />
        <b>변수양</b>
      </a>
      <br />
      <sub>🎓 인공지능학부</sub>
      <br /><br />
      <b>Research Topic</b>
      <br />
      <sub>LoRA + Spatial Reasoning </sub>
      <br /><br />
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/dks0101">
        <img src="https://github.com/ahnsubin.png" width="130px;" style="border-radius:50%"/>
        <br /><br />
        <b>안수빈</b>
      </a>
      <br />
      <sub>🎓 인공지능학부</sub>
      <br /><br />
      <b>Research Topic</b>
      <br />
      <sub>2D RoPE-Mixed Module</sub>
      <br /><br />
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/JOONHOGITHUB">
        <img src="https://github.com/JOONHOGITHUB.png" width="130px;" style="border-radius:50%"/>
        <br /><br />
        <b>이준호</b>
      </a>
      <br />
      <sub>🎓 인공지능학부</sub>
      <br /><br />
      <b>Research Topic</b>
      <br />
      <sub>Dual-Path Aerial Adpater(DPAA)</sub>
      <br /><br />
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/iiharii">
        <img src="https://github.com/iiharii.png" width="130px;" style="border-radius:50%"/>
        <br /><br />
        <b>이하리</b>
      </a>
      <br />
      <sub>🎓 산림환경시스템학과</sub>
      <br /><br />
      <b>Research Topic</b>
      <br />
      <sub>Gated Feature Injection Module(GFIM)</sub>
      <br /><br />
    </td>
  </tr>
  <tr>
    <td align="center" width="25%">
      <a href="https://github.com/sihaun">
        <img src="https://github.com/sihaun.png" width="130px;" style="border-radius:50%"/>
        <br /><br />
        <b>조시현</b>
      </a>
      <br />
      <sub>🎓 소프트웨어학부</sub>
      <br /><br />
      <b>Research Topic</b>
      <br />
      <sub>Role-Aware Evidence Reasoning for Visual Grounding</sub>
      <br /><br />
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/choyoungchae">
        <img src="https://github.com/choyoungchae.png" width="130px;" style="border-radius:50%"/>
        <br /><br />
        <b>조영채</b>
      </a>
      <br />
      <sub>🎓 인공지능학부</sub>
      <br /><br />
      <b>Research Topic</b>
      <br />
      <sub>조영채 연구 주제<br/>(한 줄 부제)</sub>
      <br /><br />
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/choijunghwan">
        <img src="https://github.com/choijunghwan.png" width="130px;" style="border-radius:50%"/>
        <br /><br />
        <b>최정환</b>
      </a>
      <br />
      <sub>🎓 소프트웨어학부</sub>
      <br /><br />
      <b>Research Topic</b>
      <br />
      <sub>최정환 연구 주제<br/>(한 줄 부제)</sub>
      <br /><br />
    </td>
    <td align="center" width="25%">
      <!-- 빈 칸 (4명 중 1자리) -->
    </td>
  </tr>
</table>

