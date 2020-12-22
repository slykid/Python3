package com.java.kilhyun.Basic;

public class ex09_BreakNContinue {

    /*
     * break
     * - 감싸고 있는 블록의 제어를 빠져나오는 기능
     * - 반복문, 조건문에서 주로 많이 사용됨(거의 반복문 내에서 조건문을 줘서 해당 조건이 맞으면 반복문에서 빠져나오는 경우에 많이 사용됨)
     *
     * continue
     * - 반복문을 수행중에 조건이 맞지 않는 경우 해당 부분 이하의 내용을 무시하고 다음 step으로 넘어가는 기능
     */
    public static void main(String[] args)
    {
        int sum = 0;
        int num = 0;

        // 1. break 문
        for(num = 1; ; num++)
        {
            sum += num;

            if (sum >= 100)
                // 본래 100보다 커지는 때는 num = 14 일때이며, 해당 스텝이 종료되는 순간 num에 1이 증가하여 15로 출력되는 것임
                // 때문에 만약 해당 구문이 없는 경우 num 은 15로 출력됨
                // 해당 구문은 sum이 100을 넘는 순간 반복문을 종료하기 때문에 1증가하는 것은 실행되지 않음

                break;

            else if(num%3==0)
                System.out.println(num);

        }

        System.out.println("---------------res-----------------");
        System.out.println(sum);
        System.out.println(num);
        System.out.println("------------------------------------------");

        // 2. continue 문
        num = 0; sum = 0;

        for(num = 2; num <= 9; num++)
        {
            if(num%2 == 1)
                continue;
            else
            {
                for(int i = 1; i <= 9; i++)
                    System.out.println(num + " * " + i + " = " + (num*i) );
            }
            System.out.println("------------------------------------------");

        }
    }

}
