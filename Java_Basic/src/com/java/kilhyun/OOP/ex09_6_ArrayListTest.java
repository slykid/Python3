package com.java.kilhyun.OOP;

import java.util.ArrayList;

public class ex09_6_ArrayListTest {

    /*
     * ArrayList
     * - 자바에서 제공되는 객체 배열이 구현된 클래스
     * - 사용 시에는 java.util.ArrayList 패키지를 import 해서 사용
     * - 객체 배열을 사용하는 데 필요한 여러 메소드들이 구현되어 있음
     *
     * - 주요 메소드
     *   - add()
     *   - size()
     *   - get()
     *   - remove()
     *   - isEmpty()
     */

    public static void main(String[] args)
    {
        ArrayList<String> list = new ArrayList<String>(); 	// 제너릭 (<> 안의 내용) 을 선언하지 않으면, 모든 형식의 데이터를 사용할 수 있으나
        // 이 후 요소를 사용하는 과정에서 반드시 형변환을 해줘야하는 번거로움이 존재함

        list.add("ghKim");
        list.add("slykid");

        for(int i = 0; i < list.size(); i++)
            System.out.println(list.get(i));

        System.out.println();

        // 향상된 for 문
        for(String s : list)
        {
            System.out.println(s);
        }
        System.out.println();
    }

}
