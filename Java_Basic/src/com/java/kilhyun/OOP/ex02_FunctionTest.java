package com.java.kilhyun.OOP;

public class ex02_FunctionTest {

    /*
     * 함수
     * - 하나의 기능을 수행하는 일련의 코드
     * - 호출하여 사용하고 기능이 수행된 후 값을 반환할 수도 있음
     * - 구현된 기능은 여러 곳에서 호출되어 사용가능함
     *
     * - 구조
     *   - 이름, 매개변수, 반환 값, 몸체로 구성됨
     *     - 예시
     *       int add( int num1, int num2)
     *       {
     *       	int result;
     *
     *       	result = num1 + num2;
     *
     *       	return result;
     *       }
     */

    public static int addNum(int num1, int num2)
    {
        int result;
        result = num1 + num2;

        return result;
    }

    public static int calcSum()
    {
        int sum = 0;

        for(int i = 0; i < 100; i++)
        {
            sum += i;
        }

        return sum;

    }

    public static void main(String[] args)
    {
        int num1 = 1, num2 = 2;
        int res = 0;
        int sum = 0;

        res = addNum(num1, num2);
        sum = calcSum();

        System.out.println("addNum 결과 : " + res);
        System.out.println("calcSum 합계 : " + sum);


    }

}
