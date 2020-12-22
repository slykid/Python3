package com.java.kilhyun.OOP;

/*
 * Q. 아래의 내용을 프로그래밍 하시오
 * Lee 학생은 국어, 수학 2과목을 수강하고, Kim 학생은 국어, 수학, 영어 3과목을 수강한다.
 * Lee 는 국어 100점, 수학 90점을 맞혔고, Kim 은 국어 100점, 수학 90점, 영어 80점을 맞혔다.
 * 위의 내용을 ArrayList 멤버변수 1개로 구현하고 결과를 출력하시오
 */

public class ex09_1_Array_Quiz2 {

    public static void main(String args[])
    {

        StudentSubject lee = new StudentSubject(0, "Lee");
        StudentSubject kim = new StudentSubject(1, "Kim");

        lee.addScore("국어", 100);
        lee.addScore("수학", 90);

        kim.addScore("국어", 100);
        kim.addScore("수학", 90);
        kim.addScore("영어", 80);


        lee.showStudentInfo();
        System.out.println("========================================");
        kim.showStudentInfo();


        /*
         * 학생 Lee 님의 국어과목의 점수는 100점 입니다.
         * 학생 Lee 님의 수학과목의 점수는 90점 입니다.
         * 학생 Lee 님의 총점은 190점 입니다.
         * ========================================
         * 학생 Kim 님의 국어과목의 점수는 100점 입니다.
         * 학생 Kim 님의 수학과목의 점수는 90점 입니다.
         * 학생 Kim 님의 영어과목의 점수는 80점 입니다.
         * 학생 Kim 님의 총점은 270점 입니다.
         */

    }

}
