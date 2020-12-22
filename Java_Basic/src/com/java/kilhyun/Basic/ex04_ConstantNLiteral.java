package com.java.kilhyun.Basic;

public class ex04_ConstantNLiteral {
    /*
     * 상수 (Constant) : 변하지 않는 수
     * 리터럴 (Literal) : 프로그램에서 사용하는 모든 숫자, 값, 논리 값
     *
     *  모든 리터럴은 상수 풀(Constant Pool)에 저장됨
     *  상수 풀에 저장될 시 정수는 int, 실수는 double 로 저장됨
     *
     *  형 변환 : 서로 다른 자료형의 값이 대임되는 경우 형변환이 발생함
     *
     *  묵시적 형변환(Promotion) : 작은 수에서 큰 수로, 덜 정밀한 수에서 더 정밀한 수로 대입되는 경우
     *  명시적 형변환(Casting) : 변환 되는 자료형을 명시하며, 자료의 손실이 발생할 수 있다.
     */

    public static void main(String[] args)
    {
        System.out.println("1. Promotion & Casting");

        // Promotion
        byte vByte = 10;
        int vInt = vByte;

        System.out.println(vByte + ", " + vInt);

        // Casting
        double vDouble = 2;

        // vInt = vDouble; // Type mismatch: cannot convert from double to int 에러 발생
        vInt = (int)vDouble;

        System.out.println(vDouble + ", " + vInt);

        System.out.println("\n2. Binary Test");

        // 2진수, 8진수, 10진수, 16진수
        int bNum = 0B1010;
        int oNum = 012;
        int dNum = 10;
        int xNum = 0XA;

        System.out.println(bNum);
        System.out.println(oNum);
        System.out.println(dNum);
        System.out.println(xNum);

    }

}
