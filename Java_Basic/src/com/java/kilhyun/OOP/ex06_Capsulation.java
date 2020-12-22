package com.java.kilhyun.OOP;

public class ex06_Capsulation {

    /* 접근제어자
     * - 변수, 메소드, 생성자에 대한 접근 권한을 지정
     * - public, private, protected, 기본 접근 제어자(같은 패키지 내에서만 보여짐) 가 있음
     *
     * - private 사용 시, 클래스 외부에서는 접근할 수 없음
     *
     * 정보은닉
     * - 외부에서 클래스 내부의 정보에 접근하지 못하도록 private 키워드를 활용하는 것으로,
     *   클래스 내부 데이터를 잘못 사용하는 오류를 방지할 수 있다.
     * - private 처리된 변수들은 get/set 메소드(getter, setter) 를 이용해 접근이 가능함
     *
     * getter & setter
     * - private 으로 접근 제한된 경우 해당 변수에 접근하기 위한 함수
     *
     * this
     * - 자신의 메모리를 가리킴
     * - 생성자에서 다른 생성자를 호출함
     * - 인스턴스 자신의 주소를 반환
     */

    public static void main(String args[])
    {
        MyDate myDate = new MyDate();

//		myDate.day = 21;
//		myDate.month = 3;
//		myDate.year = 2020;

        myDate.setDay(21);
        myDate.setMonth(7);
        myDate.setYear(2019);

        myDate.showDate();


        Person personKim = new Person("Kim", 28);
        personKim.showInfo();

        Person p = personKim.getSelf();

        System.out.println(personKim);
        System.out.println(p);

    }
}
