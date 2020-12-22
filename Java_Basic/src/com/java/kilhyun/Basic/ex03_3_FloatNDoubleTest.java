package com.java.kilhyun.Basic;

public class ex03_3_FloatNDoubleTest {

    // Java 에서는 기본적으로 double 로 저장됨
    // 따라서 float 형을 사용할 경우 F, f 등을 명시해줘야함.

    public static void main(String[] args)
    {
        double dNum = 3.14;
        float fNum = 3.14F;  // F, f 를 맨 마지막에 붙여줘야함.

        // 부동소수점의 오류
        // - 오차가 발생하지만 그만큼 소수점이하의 수를 표현하기 위해서...
        double input = 1;

        for(int i = 0; i < 10000; i++)
        {
            input = input + 0.1;
        }

        System.out.println(input);  // 1001.000000000159
    }

}
