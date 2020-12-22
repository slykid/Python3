package com.java.kilhyun.OOP;

public class ex10_EnumMethodTest {

    public static void main(String[] args)
    {
        Week today = Week.SUNDAY;

        // name()
        String name = today.name();
        System.out.println(name);

        // ordinal()
        int ordinal = today.ordinal();
        System.out.println(ordinal);

        // compareTo()
        Week day1 = Week.MONDAY;
        Week day2 = Week.TUESDAY;

        int result1 = day1.compareTo(day2);
        int result2 = day2.compareTo(day1);

        System.out.println(result1);
        System.out.println(result2);

        // valueOf()
        if(args.length == 1)
        {
            String strDay = args[0];
            Week weekDay = Week.valueOf(strDay);

            if(weekDay == Week.SATURDAY || weekDay == Week.SUNDAY)
            {
                System.out.println("주말 입니다.");
            }
            else
            {
                System.out.println("평일 입니다.");
            }

        }

        // values()
        Week[] days = Week.values();
        for(Week day : days)
        {
            System.out.println(day);
        }

    }
}
