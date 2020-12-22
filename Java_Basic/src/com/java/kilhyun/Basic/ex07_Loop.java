package com.java.kilhyun.Basic;

import java.util.Scanner;

public class ex07_Loop {

    /* 반복문
     * - 동일한 수행문을 조건에 맞는 동안 수행하도록 하는 기능
     *
     * while & do-while 문
     * while 은 조건문을 만족하는 경우 계속 반복적으로 수행
     * do-while 은 무조건 한번의 반복문 내 코드를 수행하고 조건이 맞다면 반복적으로 수행함
     *
     * for 문
     * while 문과 동일하게 반복문이지만, 횟수 제한을 할 수 있다는 장점이 있다.
     *
     */

    public static void main(String[] args)
    {

        int a = 1;
        int b = 0;

        // 1. while 문
        // - 형식: while(조건식)
        // - 조건식 부분에 True를 입력 시, 무한루프가 됨
        while(a <= 10)
        {
            b += a;
            System.out.print("a : " + a + "\t");
            System.out.println("b : " + b);
            a++;

        }

        // 2. do-while 문
        a = 1; b = 0;
        do
        {
            b += a;
            System.out.print("a : " + a + "\t");
            System.out.println("b : " + b);
            a++;
        }while(a <= 10);


        // 3. for문
        a = 1; b = 0;
        for(int i = 0; i < 10; i++)
        {
            b += a;
            System.out.print("a : " + a + "\t");
            System.out.println("b : " + b);
            a++;
        }

        // ex. 1~5 까지 모두 더하는 코드를 while, do-while, for 를 사용하여 구현하시오 (단, 0이 들어오면 종료한다.)
        System.out.println("-------------------------ver.while---------------------------");
        int input = 0;
        int sum = 0;

        Scanner sc = new Scanner(System.in);

        input = sc.nextInt();

        while(input != 0)
        {
            sum += input;
            input = sc.nextInt();  // while 을 사용할 경우 2번의 nextInt() 가 포함됨
        }

        System.out.println("Sum = " + sum);

        System.out.println("-------------------------ver.do-while---------------------------");
        input = 0;
        sum = 0;


        do
        {
            input = sc.nextInt();  // do-while 을 사용할 경우 2번의 nextInt() 가 포함됨
            sum += input;
        }
        while(input != 0);

        System.out.println("Sum = " + sum);

        System.out.println("-------------------------ver.for---------------------------");
        input = 0;
        sum = 0;

        for (;;) // for문으로 무한루프 구현할 경우
        {
            input = sc.nextInt();

            if(input != 0)
            {
                sum += input;
            }
            else
                break;

        }

        System.out.println("Sum = " + sum);
    }

}
