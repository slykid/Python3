package com.java.kilhyun.OOP;

public class ex01_Student {
    /*
     * 객체
     * - 의사나 행위가 미치는 대상(사전적 의미)
     * - 구체적 혹은 추상적 데이터의 단위
     *
     * 객체지향프로그래밍
     * - 객체를 기반의 하는 프로그래밍
     * - 객체를 정의하고 기능을 구현하고, 객체간의 협력관계를 구현함
     *
     * 절차지향 프로그래밍
     * - 시간 혹은 사건의 흐름에 따른 절차를 구현하는 프로그래밍
     *
     * 클래스
     * - 객체를 코드화한 것
     * - 객체 지향 프로그래밍에서 가장 기본적인 요소
     * - 객체의 청사진
     *
     * - 구성
     *   - 멤버변수 : 객체가 가지는 속성을 변수로 표현된 것
     *   - 메소드 : 객체의 기능을 구현한 것
     */

    // 학생(학번, 이름, 주소)
    public int studentId = 0;
    public String name = null;
    public String addr = null;

    public void showStudentInfo()
    {
        System.out.println("학번 : " + studentId);
        System.out.println("학생명 : " + name);
        System.out.println("주소 : " + addr);
    }

}
