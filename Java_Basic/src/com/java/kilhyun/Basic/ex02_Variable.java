package com.java.kilhyun.Basic;

public class ex02_Variable {
    /* 변수 명명 규칙
      - 영문자나 숫자를 사용하며, 숫자로 시작할 수는 없다.
      - 특수 문자는 '_' 와 '$' 만 사용가능하다.
      - 예약어는 사용할 수 없다.
      - 변수의 쓰임에 맞는 이름으로 명명한다.
        ex. numberOfStudent
     */
    public static void main(String[] args)
    {
        int age = 28;
        String name = "홍길동";

        // 변수 여러 개를 동시에 선언하는 경우에는 반드시 선언부 다음에 값을 대입해 주는 코드가 있어야한다.
        int count, a;
        count = 1;
        a = 1;

        System.out.println(name + " 님의 올해 나이는 " + age + "살 입니다.");
        System.out.println(count + " , " + a);
    }

}
