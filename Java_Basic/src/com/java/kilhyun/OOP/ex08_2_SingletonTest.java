package com.java.kilhyun.OOP;

import java.util.Calendar;

public class ex08_2_SingletonTest {

    /*
     * Singleton 패턴
     * - 프로그램 디자인 패턴 중 하나
     * - 단 하나만 존재해야되는 인스턴스의 경우에 사용되며, 생성자는 private으로 선언하고, static으로 유일한 객체를 생성함
     * - 외부에서 유일한 객체를 참조할 수 있도록 public static getter() 메소드로 구현한다.
     * - C 에서의 global 키워드 와  기능적으로 유사함
     *
     */

    public static void main(String[] args)
    {
//		Ex_SingletonCompany company = new Ex_SingletonCompany();  // private 이라서, 이런 식으로는 호출 불가능

        // Singleton 패턴으로 생성된 객체를 호출하는 방법
        ex08_2_Singleton_Company company1 = ex08_2_Singleton_Company.getInstance();
        ex08_2_Singleton_Company company2 = ex08_2_Singleton_Company.getInstance();

        // ex. java.util 의 Calendar 가 Singleton 패턴으로 생성된 객체임
        Calendar calender = Calendar.getInstance();
    }

}
