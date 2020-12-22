package com.java.kilhyun.OOP;

import java.util.Scanner;

public class ex05_Constructor {

    public static void main(String[] args)
    {
        // 변수 선언 및 초기화
        int id = 0;
        String name = "";
        String addr = "";

        Scanner sc = new Scanner(System.in);

        // 변수에 대입
        System.out.print("학번 : ");
        id = Integer.parseInt(sc.nextLine().toString());

        System.out.print("이름 : ");
        name = sc.nextLine();

        System.out.print("주소 : ");
        addr = sc.nextLine();

        System.out.println();

        // 객체 생성
        // - 입력값이 존재하는 경우 입력을 받은 후에 생성한다.
        Student student1 = new Student(id, name, addr);
        Student student2= new Student(id, name);

        // 입력 정보 출력
        student1.showStudentInfo();
        System.out.println();
        student2.showStudentInfo();
    }

}
