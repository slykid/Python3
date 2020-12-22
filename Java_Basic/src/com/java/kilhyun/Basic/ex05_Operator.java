package com.java.kilhyun.Basic;

public class ex05_Operator {

    /* 항 (append) : 연산에 사용되는 값
     *
     * 연산자 (operator) : 항을 이용해 연산하는 기호
     * - 단항연산자 : 항이 1개인 연산자	(ex. ++, --, ... )
     * - 이항연산자 : 항이 2개인 연산자  (ex. +, -, *, /, %, ...)
     * - 삼항연산자(조건연산자) : 항이 3개인 연산자  (ex. ? )
     *
     * 대입연산자 : 왼쪽 변수에 오른쪽 값을 대입함, 우선순위가 가장 낮은 연산자
     * 부호연산자 : 단항연산자이며, 변수의 부호변환을 담당하며 실제 값을 변경할 경우 대입연산자와 같이 사용
     *
     * 사칙연산자 : +(덧셈) , -(뺄셈), *(곱셈), /(나눗셈), %(나머지 연산)
     *
     * 복합 대입 연산자
     * - += : 두 항의 값을 더해 왼쪽 항에 대입
     * - -= : 왼쪽에서 오른쪽항의 값을 빼서 왼쪽 항에 대입
     * - *= : 두 항의 값을 곱해 왼쪽항에 대입
     * - /= : 왼쪽항의 값을 오른쪽항의 값으로 나눈 후 왼쪽 항에 대입
     * - %= : 왼쪽항의 값을 오른쪽항의 값으로 나눈 나머지를 왼쪽 항에 대입
     *
     * 증감 연산자
     * - 단항연상자이며 변수의 증감을 할 때 사용한다.
     * - 단, 사용 시 변수의 앞에 사용할지, 뒤에 사용할지를 정해서 사용해야한다.
     *
     * - 앞에서 사용할 경우 : 1을 증감시키고 문장의 연산을 수행
     * - 뒤에서 사용할 경우 : 문장의 연산을 수행한 후 1을 증감시킴
     *
     * 관계 연산자(비교연산자)
     * - 연산의 결과를 true/false (boolean) 으로 반환함
     * - ex. >, <, >=, <=, ==, !=
     *
     * 논리 연산자
     * - 관계 연산자와 함께 사용되는 경우가 많음
     * - 연산의 결과를 true/false (boolean) 으로 반환함
     *
     * - &&(논리곱) : 두 항이 모두 참인 경우만 결과가 참, 나머지 결과는 거짓
     * - ||(논리합) : 두 항 중 1개만 참이어도 결과는 참, 둘 다 거짓인 경우에만 결과가 거짓
     * - !(부정) : 참인 경우는 거짓으로, 거짓인 경우는 참으로 결과를 바꿈
     *
     * - Short Circuit Evaluation (단락 회로 평가)
     *   - 논리 곱은 두 항이 모두 참인 경우만 참이기 때문에, 만약 앞의 항이 false 라면 뒤 항의 결과는 평가하지 않아도 됨
     *   - 논리 합은 두 항 중 1개만 참인 경우면 참이기 때문에, 만약 앞의 항이 true 라면 뒤 항의 결과는 평가하지 않아도 됨
     *   - 하지만 실제 프로그램에서는 예상치 못한 결과가 발생할 수 있으므로 주의 !!!
     *     => 모든 항이 evaluation 되지 않을 수 있다.
     *
     * 조건 연산자
     * - 삼항 연산자로 사용형식은 아래와 같다.
     *   조건식 ? 결과1 : 결과2
     *   - 조건에 만족할 경우 결과 1을 반환, 만족하지 않으면 결과2를 반환
     *
     * 비트 연산자
     * - ~ : 비트 반전
     * - & : 비트 AND
     * - | : 비트 OR
     * - ^ : 비트 XOR (두 비트가 서로 다른 경우에만 1, 같으면 0)
     * - << : 왼쪽 shift (곱셈의 의미, 부호비트가 채워짐)
     * - >> : 오른쪽 shift (나누기의 의미, 부호비트가 채워짐)
     * - >>> : 오른쪽 shift 연산이지만, 채워지는 비트는 부호 상관 없이 0임
     *
     * - 마스크 : 특정 비트를 가리고 몇 개의 비트 값만 사용할 경우
     * - 비트켜기 : 특정 비트들만을 1로 설정해 사용할 경우
     *           ex. & 00001111 (하위 4비트 중 1인 비트만 꺼냄)
     *
     * - 비트끄기 : 특정 비트들만을 0으로 설정해 사용할 경우
     *           ex. | 11110000 (하위 4비트 중 0인 비트만 0으로 만들 경우)
     *
     * - 비트 토글 : 모든 비트들을 0 -> 1 혹은 1 -> 0으로 바꾸고 싶은 경우
     */

    public static void main(String[] args)
    {
        // 변수 선언
        int a = 10;		// 대입
        int b = 0;		// 초기화

        System.out.println("변경 전");
        System.out.println(a);
        System.out.println(b);

        b = -a;		//부호연산자 사용
        System.out.println("\n변경 후");
        System.out.println(a);
        System.out.println(b);


        System.out.println("\n사칙연산");

        a = 5;
        b = 3;

        System.out.println("덧셈:	 " + a + " + " + b + " = " + (a + b) );
        System.out.println("뻴셈:	 " + a + " - " + b + " = " + (a - b) );
        System.out.println("곱셈:	 " + a + " * " + b + " = " + (a * b) );
        System.out.println("나눗셈: " + a + " / " + b + " = " + (a / b) );
        System.out.println("나머지연산: " + a + " % " + b + " = " + (a % b) );

        System.out.println("\n복합연산");
        System.out.println("덧셈:	 " + (a += b) );
        System.out.println("뻴셈:	 " + (a -= b) );
        System.out.println("곱셈:	 " + (a *= b) );
        System.out.println("나눗셈: " + (a /= b) );
        System.out.println("나머지연산: " + (a %= b) );


        System.out.println("\n증감연산");

        a = 5;

        System.out.println("앞에서 사용할 경우 : " + (++a) + "\n");
        System.out.println("뒤에서 사용할 경우 : " + (a++));
        System.out.println("뒤에서 사용 후 : " + a);

        System.out.println("\n관계 & 논리 연산");

        a = 10;
        b = 2;

        // 논리 곱인 경우
        boolean value = ( (a = a + 10) < 10 ) && ( (b = b + 2) < 10);
        System.out.println(a);
        System.out.println(b);
        System.out.println(value);
        /*
         * 20
         * 2  // short circuit evaluation 발생
         * false
         */

        // 논리 합인 경우
        System.out.println("\n");
        value = ( (a = a + 10) < 10 ) || ( (b = b + 2) < 10);
        System.out.println(a);
        System.out.println(b);
        System.out.println(value);
        /*
         * 30
         * 4
         * true
         */

        System.out.println("\n조건연산");

        a = 10;
        b = 20;

        int max = (a > b) ? a : b;

        System.out.println(max);

        System.out.println("\n비트연산");

        a = 0B00001010;  // 10
        b = 0B00000101;  // 5

        System.out.println(a & b);
        /*
         * 00001010
         * 00000101
         * ---------
         * 00000000
         */

        System.out.println(a | b);
        /*
         * 00001010
         * 00000101
         * ---------
         * 00001111
         */

        System.out.println(a ^ b);
        /*
         * 00001010
         * 00000101
         * ---------
         * 00001111
         */

        System.out.println(b <<= 1);
        // 00000101 -> 00001010 (5 x 2 = 10)

        System.out.println(b >>= 1);
        // 00001010 -> 00000101 (10 / 2 = 10)
    }

}
